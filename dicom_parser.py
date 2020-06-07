from dicompylercore import dicomparser
import numpy as np
import os
import pydicom
from dicompylercore import dvhcalc
from tqdm import tqdm
from skimage import draw
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import argparse
import glob


class PatientDataLoader:
    def __init__(self, path):
        self.path = path
        self.filearray = []
        self.parse_patient = None
        self.patient = None
        for root, dirs, filenames in os.walk(path):
            for f in filenames:
                if f.endswith(".dcm"):
                    self.filearray.append(os.path.join(root, f))

    def start_parsing(self, rxdose=740, output_root=None):
        self.get_patient_data(rxdose)
        self.parse_patient_data()
        self.get_patient_images()
        self.get_structure_and_isodose(output_root)

    def get_patient_data(self, rxdose):
        """Get the data of the selected patient from the DICOM importer dialog."""
        for n in range(0, len(self.filearray)):
            dcmfile = str(os.path.join(self.path, self.filearray[n]))
            dp = dicomparser.DicomParser(dcmfile)
            if (n == 0):
                self.patient = {}
                self.patient['rxdose'] = rxdose
            if (('ImageOrientationPatient' in dp.ds) and not (dp.GetSOPClassUID() == 'rtdose')):
                if not 'images' in self.patient:
                    self.patient['images'] = []
                self.patient['images'].append(dp.ds)
            elif (dp.ds.Modality in ['RTSTRUCT']):
                self.patient['rtss'] = dp.ds
            elif (dp.ds.Modality in ['RTPLAN']):
                self.patient['rtplan'] = dp.ds
            elif (dp.ds.Modality in ['RTDOSE']):
                self.patient['rtdose'] = dp.ds
        # Sort the images based on a sort descriptor:
        # (ImagePositionPatient, InstanceNumber or AcquisitionNumber)
        if 'images' in self.patient:
            sortedimages = []
            unsortednums = []
            sortednums = []
            images = self.patient['images']
            sort = 'IPP'
            # Determine if all images in the series are parallel
            # by testing for differences in ImageOrientationPatient
            parallel = True
            for i, item in enumerate(images):
                if (i > 0):
                    iop0 = np.array(item.ImageOrientationPatient)
                    iop1 = np.array(images[i - 1].ImageOrientationPatient)
                    if (np.any(np.array(np.round(iop0 - iop1),
                                        dtype=np.int32))):
                        parallel = False
                        break
                    # Also test ImagePositionPatient, as some series
                    # use the same patient position for every slice
                    ipp0 = np.array(item.ImagePositionPatient)
                    ipp1 = np.array(images[i - 1].ImagePositionPatient)
                    if not (np.any(np.array(np.round(ipp0 - ipp1),
                                            dtype=np.int32))):
                        parallel = False
                        break
            # If the images are parallel, sort by ImagePositionPatient
            if parallel:
                sort = 'IPP'
            else:
                # Otherwise sort by Instance Number
                if not (images[0].InstanceNumber == images[1].InstanceNumber):
                    sort = 'InstanceNumber'
                # Otherwise sort by Acquisition Number
                elif not (images[0].AcquisitionNumber == images[1].AcquisitionNumber):
                    sort = 'AcquisitionNumber'

            # Add the sort descriptor to a list to be sorted
            for i, image in enumerate(images):
                if (sort == 'IPP'):
                    unsortednums.append(image.ImagePositionPatient[2])
                else:
                    unsortednums.append(image.data_element(sort).value)

            # Sort image numbers in descending order for head first patients
            if ('hf' in image.PatientPosition.lower()) and (sort == 'IPP'):
                sortednums = sorted(unsortednums, reverse=True)
            # Otherwise sort image numbers in ascending order
            else:
                sortednums = sorted(unsortednums)

            # Add the images to the array based on the sorted order
            for s, slice in enumerate(sortednums):
                for i, image in enumerate(images):
                    if (sort == 'IPP'):
                        if (slice == image.ImagePositionPatient[2]):
                            sortedimages.append(image)
                    elif (slice == image.data_element(sort).value):
                        sortedimages.append(image)

            # Save the images back to the patient dictionary
            self.patient['images'] = sortedimages
            # self.parse_patient_data()
            # self.get_patient_images()

    def GetDoseGridPixelData(self, pixlut, doselut):
        """Convert dosegrid data into pixel data using the dose to pixel LUT."""

        dosedata = []
        x = []
        y = []
        # Determine if the patient is prone or supine
        imdata = self.images[self.imagenum - 1].GetImageData()
        prone = -1 if 'p' in imdata['patientposition'].lower() else 1
        feetfirst = -1 if 'ff' in imdata['patientposition'].lower() else 1
        # Get the pixel spacing
        spacing = imdata['pixelspacing']

        # Transpose the dose grid LUT onto the image grid LUT
        x = (np.array(doselut[0]) - pixlut[0][0]) * prone * feetfirst / spacing[0]
        y = (np.array(doselut[1]) - pixlut[1][0]) * prone / spacing[1]
        return (x, y)

    def GetContourPixelData(self, pixlut, contour, prone=False, feetfirst=False):
        """Convert structure data into pixel data using the patient to pixel LUT."""

        pixeldata = []
        # For each point in the structure data
        # look up the value in the LUT and find the corresponding pixel pair
        for p, point in enumerate(contour):
            for xv, xval in enumerate(pixlut[0]):
                if (xval > point[0] and not prone and not feetfirst):
                    break
                elif (xval < point[0]):
                    if feetfirst or prone:
                        break
            for yv, yval in enumerate(pixlut[1]):
                if (yval > point[1] and not prone):
                    break
                elif (yval < point[1] and prone):
                    break
            pixeldata.append((xv, yv))
        return pixeldata

    def parse_patient_data(self):
        """Thread to load the patient data."""
        ptdata = self.patient
        patient = {}
        pbar = tqdm(total=100)
        if not 'images' in ptdata:
            # Look for DICOM data in the ptdata dictionary
            for rtdatatype in ptdata.keys():
                if isinstance(ptdata[rtdatatype], pydicom.dataset.FileDataset):
                    patient.update(dicomparser.DicomParser(ptdata[rtdatatype]).GetDemographics())
                    break
        if 'rtss' in ptdata:
            # pbar.update(20)
            pbar.n = 20
            pbar.set_description('Processing RT Structure Set...')
            pbar.refresh()

            d = dicomparser.DicomParser(ptdata['rtss'])
            s = d.GetStructures()
            for k in s.keys():
                s[k]['planes'] = d.GetStructureCoordinates(k)
                s[k]['thickness'] = d.CalculatePlaneThickness(s[k]['planes'])
            patient['structures'] = s
        if 'rtplan' in ptdata:
            pbar.n = 40
            pbar.refresh()
            pbar.set_description('Processing RT Plan...')
            patient['plan'] = dicomparser.DicomParser(ptdata['rtplan']).GetPlan()
        if 'rtdose' in ptdata:
            pbar.n = 60
            pbar.set_description('Processing RT Dose...')
            pbar.refresh()
            patient['dvhs'] = dicomparser.DicomParser(ptdata['rtdose']).GetDVHs()
            patient['dose'] = dicomparser.DicomParser(ptdata['rtdose'])
        if 'images' in ptdata:
            pbar.n = 80
            pbar.set_description('Processing Images...')
            pbar.refresh()
            if not 'id' in patient:
                patient.update(dicomparser.DicomParser(ptdata['images'][0]).GetDemographics())
            patient['images'] = []
            for image in ptdata['images']:
                patient['images'].append(dicomparser.DicomParser(image))
        if 'rxdose' in ptdata:
            if not 'plan' in patient:
                patient['plan'] = {}
            patient['plan']['rxdose'] = ptdata['rxdose']
        # if the min/max/mean dose was not present, calculate it and save it for each structure
        pbar.n = 90
        pbar.set_description('Processing DVH data...')
        pbar.refresh()
        if ('dvhs' in patient) and ('structures' in patient):
            # If the DVHs are not present, calculate them
            i = 0
            for key, structure in patient['structures'].items():
                # Only calculate DVHs if they are not present for the structure
                # or recalc all DVHs if the preference is set
                if ((not (key in patient['dvhs'].keys()))):
                    # Only calculate DVHs for structures, not applicators
                    # and only if the dose grid is present
                    if ((structure['name'].startswith('Applicator')) or (not "PixelData" in patient['dose'].ds)):
                        continue
                    pbar.n = int(np.round(10 * i / len(patient['structures']))) + 90
                    pbar.set_description('Calculating DVH for ' + structure['name'] + '...')
                    pbar.refresh()
                    # Limit DVH bins to 500 Gy due to high doses in brachy
                    dvh = dvhcalc.get_dvh(ptdata['rtss'], patient['dose'].ds, key, 50000)
                    if len(dvh.counts):
                        patient['dvhs'][key] = dvh
                    i += 1
            for key, dvh in patient['dvhs'].items():
                dvh.rx_dose = patient['plan']['rxdose'] / 100
        pbar.n = 100
        pbar.set_description('Done')
        pbar.close()
        self.parse_patient = patient

    def get_patient_images(self):
        self.z = 0
        self.structurepixlut = ([], [])
        self.dosepixlut = ([], [])
        if 'images' in self.parse_patient:
            self.images = self.parse_patient['images']
            self.imagenum = 1
            # If more than one image, set first image to middle of the series
            if (len(self.images) > 1):
                self.imagenum = int(len(self.images) / 2)
            image = self.images[self.imagenum - 1]
            self.structurepixlut = image.GetPatientToPixelLUT()
            # Determine the default window and level of the series
            self.window, self.level = image.GetDefaultImageWindowLevel()
            # Dose display depends on whether we have images loaded or not
            self.isodoses = {}
            if ('dose' in self.parse_patient and ("PixelData" in self.parse_patient['dose'].ds)):
                self.dose = self.parse_patient['dose']
                self.dosedata = self.dose.GetDoseData()
                # First get the dose grid LUT
                doselut = self.dose.GetPatientToPixelLUT()
                # Then convert dose grid LUT into an image pixel LUT
                self.dosepixlut = self.GetDoseGridPixelData(self.structurepixlut, doselut)
            else:
                self.dose = []
            if 'plan' in self.parse_patient:
                self.rxdose = self.parse_patient['plan']['rxdose']
            else:
                self.rxdose = 0
        else:
            self.images = []

    def get_structure_and_isodose(self, output_root=None):
        if output_root is None:
            output_root = os.path.join(self.path, "data")
        output_label_root = os.path.join(output_root, "mask")
        output_image_root = os.path.join(output_root, "image")
        output_dose_root = os.path.join(output_root, "dose")
        if not os.path.exists(output_root):
            os.mkdir(output_root)
        if not os.path.exists(output_label_root):
            os.mkdir(output_label_root)
        if not os.path.exists(output_image_root):
            os.mkdir(output_image_root)
        if not os.path.exists(output_dose_root):
            os.mkdir(output_dose_root)
        test_image = self.images[0].GetImage(self.window, self.level)
        dose_3D = np.zeros((len(self.images), len(self.structurepixlut[0]), len(self.structurepixlut[1])),
                           dtype=np.float32)
        image_3D = np.zeros((len(self.images), test_image.size[0], test_image.size[1]), dtype=np.uint8)
        structures = self.parse_patient["structures"]
        cmap = np.zeros((len(structures.keys()) + 1, 3), dtype=np.uint8)
        pbar = tqdm(total=len(self.images))
        for idx, image in enumerate(self.images):
            pbar.update(1)
            # 将CT slice转换成PIL.Image
            pil_image = image.GetImage(self.window, self.level)
            pil_image.save(os.path.join(output_image_root, (str(idx) + ".png")))
            image_3D[idx, :] = np.array(pil_image)
            size = pil_image.size
            imdata = image.GetImageData()
            position = '%.2f' % imdata['position'][2]
            # Determine whether the patient is prone or supine
            if 'p' in imdata['patientposition'].lower():
                prone = True
            else:
                prone = False
            # Determine whether the patient is feet first or head first
            if 'ff' in imdata['patientposition'].lower():
                feetfirst = True
            else:
                feetfirst = False
            # 绘制标签图
            pbar.set_description('Processing structure on index [' + str(idx) + ']...')
            for label, structure in structures.items():
                cmap[label, :] = structure['color']
                # Create an indexing array of z positions of the structure data
                # to compare with the image z position
                if not "zarray" in structure:
                    structure['zarray'] = np.array(
                        list(structure['planes'].keys()), dtype=np.float32)
                    structure['zkeys'] = structure['planes'].keys()
                # Return if there are no z positions in the structure data
                if not len(structure['zarray']):
                    continue
                else:
                    output_label_dir = os.path.join(output_label_root, str(label))
                    if not os.path.exists(output_label_dir):
                        os.mkdir(output_label_dir)
                # Determine the closest z plane to the given position
                zmin = np.amin(np.abs(structure['zarray'] - float(position)))
                index = np.argmin(np.abs(structure['zarray'] - float(position)))
                # Draw the structure only if the structure has contours
                # on the closest plane, within a threshold
                label_img = np.zeros(size, dtype=np.uint8)
                if (zmin < 0.5):
                    for contour in structure['planes'][list(structure['zkeys'])[index]]:
                        if (contour['type'] == u"CLOSED_PLANAR"):
                            # Convert the structure data to pixel data
                            pixeldata = np.array(self.GetContourPixelData(
                                self.structurepixlut, contour['data'], prone, feetfirst))

                            X = pixeldata[:, 0]
                            Y = pixeldata[:, 1]
                            rr, cc = draw.polygon(Y, X)
                            draw.set_color(label_img, [rr, cc], label)
                img = Image.fromarray(label_img)
                img.putpalette(cmap)
                img.save(os.path.join(output_label_dir, (str(idx) + "-" + str(label) + ".png")))
            pbar.set_description('Processing dose on index [' + str(idx) + ']...')
            # 绘制剂量图
            for xpos in range(len(self.structurepixlut[0])):
                for ypos in range(len(self.structurepixlut[1])):
                    # Lookup the current dose plane and find the value of the current
                    # pixel, if the dose has been loaded
                    if not (self.dose == []):
                        xdpos = np.argmin(np.fabs(np.array(self.dosepixlut[0]) - xpos))
                        ydpos = np.argmin(np.fabs(np.array(self.dosepixlut[1]) - ypos))
                        dosegrid = self.dose.GetDoseGrid(float(position))
                        if not (dosegrid == []):
                            dose_3D[idx, xpos, ypos] = dosegrid[ydpos, xdpos] * self.dosedata['dosegridscaling']

        pbar.set_description('Saving...')
        with open(os.path.join(output_dose_root, 'dose.pkl'), 'wb') as f:
            pickle.dump(dose_3D, f)
        with open(os.path.join(output_image_root, 'image.pkl'), 'wb') as f:
            pickle.dump(image_3D, f)
        for label in structures.keys():
            output_label_dir = os.path.join(output_label_root, str(label))
            if not os.path.exists(output_label_dir):
                continue
            mask_3D = np.zeros((len(self.images), test_image.size[0], test_image.size[1]), dtype=np.uint8)
            for idx, mask_path in enumerate(sorted(glob.glob(output_label_dir + '/*.png'),
                                                   key=lambda x: int(os.path.split(x)[-1].split('-')[0]))):
                mask_img = Image.open(os.path.join(output_label_dir, mask_path))
                mask_3D[idx, :] = np.array(mask_img)
            with open(os.path.join(output_label_dir, str(label) + '.pkl'), 'wb') as f:
                pickle.dump(mask_3D, f)
        pbar.set_description('Done')
        pbar.close()


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='input directory', default=r'F:\tools\dicom\CT\CT')
parser.add_argument('-o', '--output', type=str, help='output directory', default=r'F:\tools\dicom\CT\output')
args = parser.parse_args()
if __name__ == '__main__':
    if args.input is not None and os.path.exists(args.input):
        pdl = PatientDataLoader(args.input)
        pdl.start_parsing(output_root=args.input if args.output is None else args.output)
    else:
        raise Exception("Input/Output directory does not exist.")