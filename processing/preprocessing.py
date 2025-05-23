import mne
import numpy as np
from collections import defaultdict


class BCIDataProcessor:
    def __init__(self, recording_path, l_freq=None, h_freq=None, window_size=2, window_overlap=0.5, rescale=True, filter_method='iir', use_notch=False):
        self.recording_path = recording_path
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.rescale = rescale
        self.filter_method = filter_method
        self.use_notch = use_notch
        self.raw = None
        self.label_onsets = defaultdict(list)
        self.stimulus_duration = None
        self.cropped = None
        self.filtered = None
        self.epochs = None
        self.data_arrays = None

    def load_data(self):
        self.raw = mne.io.read_raw_fif(self.recording_path, preload=True)
        if self.rescale:
            self.raw.rescale(1e-6)
        montage = mne.channels.make_standard_montage("standard_1020")
        self.raw.info = mne.create_info(ch_names=self.raw.ch_names, sfreq=self.raw.info['sfreq'], ch_types='eeg')
        self.raw.set_montage(montage)

    def extract_annotations(self):
        for i in range(len(self.raw.annotations) - 1):
            if self.raw.annotations[i]['description'].startswith('stimulus'):
                assert self.raw.annotations[i + 1]['description'].startswith('cue')
                self.stimulus_duration = int(self.raw.annotations[i + 1]['onset'] - self.raw.annotations[i]['onset'])
                break

        for annotation in self.raw.annotations:
            if annotation['description'].startswith('cue'):
                current_cue = annotation['description'].split(' ')[1]
            elif annotation['description'].startswith('stimulus'):
                current_onset = annotation['onset']
                self.label_onsets[current_cue].append(current_onset)

    def crop_raw_data(self):
        self.cropped = {}
        for label, onsets in self.label_onsets.items():
            self.cropped[label] = []
            for onset in onsets:
                end = onset + self.stimulus_duration
                self.cropped[label].append(self.raw.copy().crop(onset, end))

    def filter_cropped_data(self):
        self.filtered = {}
        for label, raw_list in self.cropped.items():
            self.filtered[label] = []
            for raw in raw_list:
                filtered = raw.copy().filter(self.l_freq, self.h_freq, method=self.filter_method)
                if self.use_notch:
                    filtered.notch_filter(freqs=[50])
                self.filtered[label].append(filtered)

    def epoch_filtered_data(self):
        self.epochs = {}
        for label, raw_list in self.filtered.items():
            self.epochs[label] = []
            for raw in raw_list:
                epochs = mne.make_fixed_length_epochs(raw, duration=self.window_size, overlap=self.window_overlap,
                                                      preload=True)
                self.epochs[label].append(epochs)

    def convert_epochs_to_array(self):
        self.data_arrays = {}
        for label, epochs_list in self.epochs.items():
            self.data_arrays[label] = []
            for raw in epochs_list:
                self.data_arrays[label].append(raw.get_data())
            self.data_arrays[label] = np.concatenate(self.data_arrays[label], axis=0)

    def process(self, psds=False, nfft=250, channels=(0, 1, 2, 3, 4, 5, 6, 7)):
        self.load_data()
        self.extract_annotations()
        self.crop_raw_data()
        self.filter_cropped_data()
        self.epoch_filtered_data()
        self.convert_epochs_to_array()
        # Select channels
        for label, data_array in self.data_arrays.items():
            self.data_arrays[label] = data_array[:, channels, :]
        if psds:
            self.convert_arrays_to_psds(nfft)
        return self.data_arrays

    def process_chunk(self, chunk):
        """
        Does the same processing steps as process() but for a single chunk of data.
        It is assumed that n_times is equal to the window_size.
        :param chunk: (n_channels, n_times)
        :return: data_array: (1, n_channels, n_times)
        """
        # Create raw
        sfreq = self.raw.info['sfreq']
        ch_names = self.raw.ch_names
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(chunk, info)
        if self.rescale:
            raw.rescale(1e-6)

        # Apply filtering
        raw_filtered = raw.filter(self.l_freq, self.h_freq, method=self.filter_method)
        if self.use_notch:
            raw_filtered.notch_filter(freqs=[50])

        # Reshape data
        data_array = raw_filtered.get_data()
        out = data_array[np.newaxis, :, :]
        return out

    def convert_arrays_to_psds(self, nfft):
        for label, data_array in self.data_arrays.items():
            psds = []
            # Compute the PSD for each epoch
            for i in range(data_array.shape[0]):
                psd, freqs = mne.time_frequency.psd_array_welch(data_array[i], self.raw.info['sfreq'], fmin=self.l_freq,
                                                                fmax=self.h_freq, n_fft=nfft)
                psds.append(psd)
            self.data_arrays[label] = np.array(psds)
            # self.data_arrays[label] = np.mean(self.data_arrays[label], axis=0)
        return self.data_arrays


if __name__ == "__main__":
    recording_path = 'recordings/recording_imagery1_gel.raw.fif'
    processor = BCIDataProcessor(recording_path, l_freq=7, h_freq=30, window_size=2, window_overlap=0.5)
    data_arrays = processor.process()
    print(data_arrays)
