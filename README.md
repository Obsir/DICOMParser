DICOMParser
============
![](https://img.shields.io/badge/python-3.6%2B-brightgreen)

This code can read dose data, mask and image from DICOM files.

## Installation

```python
pip install -r requirements.txt
```

## Usage:

```python
python dicom_parser.py -i <intput_dir> -o <output_dir default=input_dir>
```

### Output:

	<output_dir>
		└─data
		    ├─dose
			│	└─dose.pkl 	----- 3D Dose data
		    ├─image			----- Image directory
			│	├─0.png 	
			│	├─1.png
			│	├─2.png
			│	├─...
			│	└─image.pkl ----- 3D image
		    └─mask			----- Mask directory
		        ├─1			
				│ ├─0-1.png	
				│ ├─1-1.png
				│ ├─2-1.png
				│ ├─...
				│ └─1.pkl	----- 3D mask (label 1)
		        ├─2			
				│ ├─0-2.png	
				│ ├─1-2.png
				│ ├─2-2.png
				│ ├─...
				│ └─2.pkl	----- 3D mask (label 2)
				├─...
				...

## Acknowledgments

This repository heavily borrows from「[bastula/dicompyler](https://github.com/bastula/dicompyler)」.