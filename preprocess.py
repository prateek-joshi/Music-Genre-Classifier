import os
import librosa
import math
import json

os.chdir('/content/Music-Genre-Classification/')

DATASET_PATH = 'genres'
JSON_PATH = 'data.json'
SAMPLE_RATE = 22050
DURATION = 30 # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
  """
    High-level function to save mel coefficients.
    Params:
      dataset_path - path to parent directory containing all labeled tracks
      json_path - path to json file containing configuration information
      n_mfcc - number of mel coefficients to find (default 13)
      n_fft - window size of short-time Fourier transform (default 2048)
      hop_length - window slide offset (default 512)
      num_segments - Number of segments each track is to be divided into (default 5)
  """
  # dictionary to store data
  data = {
    'mapping': [],
    'mfcc': [],
    'labels': []
  }

  num_samples_per_segment = int(SAMPLES_PER_TRACK/num_segments)
  expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment/hop_length)

  # loop through all the genres
  for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
    # ensure to be not at the root directory
    if dirpath is not dataset_path:
      # save the label
      label = dirpath.split('/')[-1]   # genre/blues -> ()
      data['mapping'].append(label)
      print(f'\nProcessing label {label}...')

      # go through the files in the genre
      for f in filenames:
        filepath = os.path.join(dirpath, f)
        signal, sr = librosa.load(filepath, sr=SAMPLE_RATE)

        # process segments, extract mfcc and store data
        for s in range(num_segments):
          start_sample = s * num_samples_per_segment
          finish_sample = start_sample + num_segments

          mfcc = librosa.feature.mfcc(
              signal[start_sample:finish_sample], 
              sr=SAMPLE_RATE, 
              n_fft=n_fft,
              hop_length=hop_length
          ).T # transpose

          data['mfcc'].append(mfcc.tolist())
          data['labels'].append(i-1)    # i-1 as we ignored the first itrn
          print(f'{filepath}, segment: {s+1}')

  with open(JSON_PATH, 'w') as fp:
    json.dump(data, fp, indent=4)


if __name__=='__main__':
  # print(os.getcwd())
  # os.chdir('/content')
  # print(os.getcwd())
  save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)










