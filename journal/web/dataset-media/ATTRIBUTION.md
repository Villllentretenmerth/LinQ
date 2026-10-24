# Educational dataset excerpts

These media are separate from the software licence. No commercial rights are granted by this repository.

## UR Fall Detection (fall-01 … fall-10)

Source: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
Authors: Bogdan Kwolek, Michal Kepski.
Citation: Human fall detection on embedded platform using depth maps and wireless accelerometer,
Computer Methods and Programs in Biomedicine 117(3), 2014, pp. 489–501.
Licence: Creative Commons Attribution-NonCommercial-ShareAlike 4.0,
https://creativecommons.org/licenses/by-nc-sa/4.0/
Adaptations distributed here under the same licence: H.264 re-encoding, removal of audio,
and JPEG preview extraction. Each recording is a different source sequence, camera 0.

## KTH Actions (run-01 … run-10)

Source and use terms: https://www.csc.kth.se/cvap/actions/
Authors: Christian Schuldt, Ivan Laptev, Barbara Caputo.
Citation: Recognizing Human Actions: A Local SVM Approach, ICPR 2004, Cambridge, UK.
The source permits non-commercial use. Original recordings: person01 … person10,
running, scenario d1. Converted from AVI to H.264 MP4 without audio; preview JPEGs extracted.

The manifest records original filenames, source URLs, durations and SHA-256 of each output video.
These are manually selected educational examples, not detection results or workplace incident evidence.
To reproduce: install core requirements, then from journal run
`python -m scripts.prepare_dataset_demo --cache /path/to/download-cache`.

## Team CP3 (cp3-*)

Source: repository docs/demo_cp3.mp4, supplied by the user. Six distinct excerpts
retain the original titles and annotations. Two are explicitly safe control actions.
Trim intervals are in manifest.json; MP4 re-encoded, audio removed, JPEG previews extracted.
These clips are team demonstration material, not additional UR Fall/KTH recordings.
