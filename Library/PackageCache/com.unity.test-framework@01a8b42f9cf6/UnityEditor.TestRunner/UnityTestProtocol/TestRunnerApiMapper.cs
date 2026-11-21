using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Xml;
using UnityEditor.TestTools.TestRunner.Api;
using UnityEditor.TestTools.TestRunner.GUI;

namespace UnityEditor.TestTools.TestRunner.UnityTestProtocol
{
    internal class TestRunnerApiMapper : ITestRunnerApiMapper
    {
        internal IGuiHelper guiHelper =  new GuiHelper(new MonoCecilHelper(), new AssetsDatabaseHelper());
        private readonly string _projectRepoPath;
        // Position of each span key in _reloadState.iterationCounters
        private readonly Dictionary<string, int> _iterationCounterIndex = new Dictionary<string, int>();

        // Bridges the spanId from TestStarted to TestFinished. The Begin-side id uses the iteration counter
        // (incremented per TestStarted), not RetryIteration, which stays 0 for skipped/inconclusive re-runs
        // and would diverge. Only leaf tests use this bridge — editor tests run serially, so a single leaf
        // is in flight at a time, which is why the bridge is a single in-flight span rather than a collection.
        // (Suites emitted by UtpMessageReporter get pure-deterministic ids — DeterministicGuid(suiteKey, 0,
        // salt) on both Begin and End, since a suite is entered exactly once per run — so Begin == End
        // reconciles without a bridge and suites never enter _inFlightSpan. The single-field invariant holds.)
        //
        // A domain reload (e.g. [InPlayMode] entering play mode) wipes in-memory state, so both are persisted
        // via _stateStore (a ScriptableSingleton, the test runner's standard reload-surviving mechanism): the
        // in-flight span so the post-reload End reuses its Begin's id, and the iteration counters so each retry's
        // Begin keeps a distinct id instead of recomputing the same one (duplicate Begins render nested in the
        // consumer; see UTR-1268).
        private SpanIdEntry _inFlightSpan;

        // Salt that keeps spanIds distinct across the editor runs of one UTR session; without it, tests
        // with identical FullNames in different packages (e.g. PackageIsolationTests) collide and the
        // consumer merges their trees. It is per editor LAUNCH, not per run, which suffices only because
        // an automated run owns its editor process — the runner starts one per rerun and kills it before
        // the next, and UTP is emitted only under -automated -runTests. Two runs sharing one launch would
        // repeat ids, since a plan message resets the iteration counters; separating those needs a run
        // identity supplied by UTR (as the native runner does with -utrRunIndex), not a launch-local one.
        private readonly string _runSalt;

        internal const string LaunchSaltKey = "UnityTestProtocol.RunSalt";

        // Identifies this editor launch. SessionState is native editor state, so it outlives the managed
        // domain reload that wipes everything above and dies with the process — the lifetime a spanId
        // needs. A process id was used here before and is not safe: the OS reuses the id of an exited
        // process, so one session's first and last editor can share one and hash their two projects
        // identically (UTR-1336).
        private static string LaunchSalt()
        {
            var salt = SessionState.GetString(LaunchSaltKey, null);
            if (string.IsNullOrEmpty(salt))
            {
                salt = Guid.NewGuid().ToString("N");
                SessionState.SetString(LaunchSaltKey, salt);
            }

            return salt;
        }

        // The launch identifies the run; the project path keeps two projects apart even if they ever
        // share one launch, and is what lets a test observe that separation in a single process. Composed
        // here alone so a test recomputing an expected spanId cannot drift from what the mapper used.
        internal static string ComposeSalt(string projectRepoPath) =>
            $"{LaunchSalt()}:{projectRepoPath ?? string.Empty}";

        private readonly ISpanIdStateStore _stateStore;

        // The iteration counters live in this persisted state directly, so saving does not copy them per test.
        private readonly SpanIdReloadState _reloadState;

        public TestRunnerApiMapper(string projectRepoPath, ISpanIdStateStore stateStore = null)
        {
            _projectRepoPath = projectRepoPath;
            _runSalt = ComposeSalt(projectRepoPath);
            _stateStore = stateStore ?? new ScriptableSingletonSpanIdStore();
            _reloadState = _stateStore.Load() ?? new SpanIdReloadState();
            LoadState();
        }

        public TestPlanMessage MapTestToTestPlanMessage(ITestAdaptor testsToRun)
        {
            _iterationCounterIndex.Clear();
            _reloadState.iterationCounters.Clear();
            _inFlightSpan = default;
            SaveState();
            var testsNames = testsToRun != null ? FlattenTestNames(testsToRun) : new List<string>();

            var msg = new TestPlanMessage
            {
                tests = testsNames
            };

            return msg;
        }

        public TestStartedMessage MapTestToTestStartedMessage(ITestAdaptor test)
        {
            var spanKey = test.UniqueName ?? test.FullName;
            var iteration = IncrementIterationCounter(spanKey);

            var spanId = DeterministicGuid.Create(spanKey, iteration, _runSalt);
            _inFlightSpan = new SpanIdEntry { spanKey = spanKey, spanId = spanId, iteration = iteration };
            SaveState();

            return new TestStartedMessage
            {
                name = test.FullName,
                spanId = spanId
            };
        }

        public TestFinishedMessage TestResultToTestFinishedMessage(ITestResultAdaptor result)
        {
            string filePathString = default;
            int lineNumber = default;
            if (result.Test.Method != null && result.Test.TypeInfo != null)
            {
                var method = result.Test.Method.MethodInfo;
                var type = result.Test.TypeInfo.Type;
                var fileOpenInfo = guiHelper.GetFileOpenInfo(type, method);
                filePathString = !string.IsNullOrEmpty(_projectRepoPath) ? Path.Combine(_projectRepoPath, fileOpenInfo.FilePath) : fileOpenInfo.FilePath;
                lineNumber = fileOpenInfo.LineNumber;
            }

            var iteration = 0;
            if(result is TestResultAdaptor)
            {
                var adaptor = ((TestResultAdaptor)result);
                iteration = adaptor.RepeatIteration == 0 ? adaptor.RetryIteration : adaptor.RepeatIteration;
            }

            var spanKey = result.Test.UniqueName ?? result.Test.FullName;
            string spanId;
            if (_inFlightSpan.spanKey == spanKey)
            {
                spanId = _inFlightSpan.spanId;
                iteration = _inFlightSpan.iteration;
                _inFlightSpan = default;
                SaveState();
            }
            else
                spanId = DeterministicGuid.Create(spanKey, iteration, _runSalt);

            return new TestFinishedMessage
            {
                name = result.Test.FullName,
                duration = Convert.ToUInt64(result.Duration * 1000),
                durationMicroseconds = Convert.ToUInt64(result.Duration * 1000000),
                message = result.Message,
                state = GetTestStateFromResult(result),
                stackTrace = result.StackTrace,
                fileName = filePathString,
                lineNumber = lineNumber,
                iteration = iteration,
                spanId = spanId
            };
        }

        public string GetRunStateFromResultNunitXml(ITestResultAdaptor result)
        {
            var doc = new XmlDocument();
            doc.LoadXml(result.ToXml().OuterXml);
            return doc.FirstChild.Attributes["runstate"].Value;
        }

        public TestState GetTestStateFromResult(ITestResultAdaptor result)
        {
            var state = TestState.Failure;

            if (result.TestStatus == TestStatus.Passed)
            {
                state = TestState.Success;
            }
            else if (result.TestStatus == TestStatus.Skipped)
            {
                state = TestState.Skipped;

                if (result.ResultState.ToLowerInvariant().EndsWith("ignored"))
                {
                    state = TestState.Ignored;
                }
            }
            else
            {
                if (result.ResultState.ToLowerInvariant().Equals("inconclusive"))
                {
                    state = TestState.Inconclusive;
                }

                if (result.ResultState.ToLowerInvariant().EndsWith("cancelled") ||
                    result.ResultState.ToLowerInvariant().EndsWith("error"))
                {
                    state = TestState.Error;
                }
            }

            return state;
        }

        public List<string> FlattenTestNames(ITestAdaptor test)
        {
            var results = new List<string>();

            if (!test.IsSuite)
                results.Add(test.FullName);

            if (test.Children != null && test.Children.Any())
                foreach (var child in test.Children)
                    results.AddRange(FlattenTestNames(child));

            return results;
        }

        // Returns the count before this increment
        private int IncrementIterationCounter(string spanKey)
        {
            var counters = _reloadState.iterationCounters;
            if (!_iterationCounterIndex.TryGetValue(spanKey, out var index))
            {
                _iterationCounterIndex[spanKey] = counters.Count;
                counters.Add(new IterationEntry { spanKey = spanKey, count = 1 });
                return 0;
            }

            var count = counters[index].count;
            counters[index] = new IterationEntry { spanKey = spanKey, count = count + 1 };
            return count;
        }

        private void LoadState()
        {
            _inFlightSpan = _reloadState.inFlightSpan;
            if (_reloadState.iterationCounters == null)
                _reloadState.iterationCounters = new List<IterationEntry>();

            _iterationCounterIndex.Clear();
            for (var i = 0; i < _reloadState.iterationCounters.Count; ++i)
                _iterationCounterIndex[_reloadState.iterationCounters[i].spanKey] = i;
        }

        private void SaveState()
        {
            _reloadState.inFlightSpan = _inFlightSpan;
            _stateStore.Save(_reloadState);
        }
    }

    // Persists the reload-surviving state. Abstracted so tests can supply an in-memory store.
    internal interface ISpanIdStateStore
    {
        SpanIdReloadState Load();
        void Save(SpanIdReloadState state);
    }

    // Unity can't serialize a Dictionary, so the counters are stored as an entry list.
    [Serializable]
    internal class SpanIdReloadState
    {
        public SpanIdEntry inFlightSpan;
        public List<IterationEntry> iterationCounters = new List<IterationEntry>();
    }

    [Serializable]
    internal struct SpanIdEntry
    {
        public string spanKey;
        public string spanId;
        public int iteration;
    }

    [Serializable]
    internal struct IterationEntry
    {
        public string spanKey;
        public int count;
    }

    // ScriptableSingleton survives domain reloads (cf. CallbacksHolder, RunData), rehydrating [SerializeField] state.
    internal class ScriptableSingletonSpanIdStore : ISpanIdStateStore
    {
        public SpanIdReloadState Load() => SpanIdStateHolder.instance.State;

        public void Save(SpanIdReloadState state) => SpanIdStateHolder.instance.State = state;
    }

    internal class SpanIdStateHolder : ScriptableSingleton<SpanIdStateHolder>
    {
        [UnityEngine.SerializeField]
        private SpanIdReloadState _state = new SpanIdReloadState();

        public SpanIdReloadState State
        {
            get => _state;
            set => _state = value;
        }
    }
}
