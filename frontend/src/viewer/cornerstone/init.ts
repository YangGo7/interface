import * as cornerstone from '@cornerstonejs/core';
import * as cornerstoneTools from '@cornerstonejs/tools';
import cornerstoneDICOMImageLoader from '@cornerstonejs/dicom-image-loader';
import dicomParser from 'dicom-parser';
import { registerMultiframeImageLoader } from './multiframeLoader';
import {
  DicomOverlayMetadata,
  parseDicomMetadataFromDataSet,
  parseDicomNumber,
  parseDicomNumberList,
} from './dicomMetadata';

let initPromise: Promise<void> | null = null;
const localDicomFiles = new Map<string, File>();
const localDicomMetadata = new Map<string, DicomOverlayMetadata>();

export function registerNativeDicomFile(file: File) {
  const imageId = cornerstoneDICOMImageLoader.wadouri.fileManager.add(file);
  if (!localDicomMetadata.has(imageId)) {
    void cacheDicomMetadata(imageId, file);
  }
  return imageId;
}

export function registerLocalDicomFile(key: string, file: File) {
  const imageId = `dicomlocal:${key}`;
  localDicomFiles.set(imageId, file);
  localDicomMetadata.delete(imageId);
  void cacheDicomMetadata(imageId, file);
  return imageId;
}

async function cacheDicomMetadata(imageId: string, file: File) {
  if (localDicomMetadata.has(imageId)) {
    return localDicomMetadata.get(imageId)!;
  }

  const dataSet = dicomParser.parseDicom(new Uint8Array(await file.arrayBuffer()), { untilTag: 'x7fe00010' });
  const metadata = parseDicomMetadataFromDataSet(dataSet, file.name);
  localDicomMetadata.set(imageId, metadata);
  return metadata;
}

export async function registerLocalDicomFileWithMetadata(key: string, file: File) {
  const imageId = registerLocalDicomFile(key, file);
  await cacheDicomMetadata(imageId, file);
  return imageId;
}

export function getRegisteredDicomMetadata(imageId?: string | null) {
  return imageId ? localDicomMetadata.get(imageId) || null : null;
}

export async function registerNativeDicomFileWithMetadata(file: File) {
  const imageId = registerNativeDicomFile(file);
  await cacheDicomMetadata(imageId, file);
  return imageId;
}

const getMinMax = (scalarData: ArrayLike<number>) => {
  let minPixelValue = Number.POSITIVE_INFINITY;
  let maxPixelValue = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < scalarData.length; index += 1) {
    const value = scalarData[index];
    if (value < minPixelValue) {
      minPixelValue = value;
    }
    if (value > maxPixelValue) {
      maxPixelValue = value;
    }
  }

  if (!Number.isFinite(minPixelValue) || !Number.isFinite(maxPixelValue)) {
    return { minPixelValue: 0, maxPixelValue: 0 };
  }

  return { minPixelValue, maxPixelValue };
};

const webImageLoader = (imageId: string): cornerstone.Types.IImageLoadObject => {
  const url = imageId.startsWith('web:') ? imageId.substring(4) : imageId;

  const promise = new Promise<cornerstone.Types.IImage>((resolve, reject) => {
    const image = new Image();
    if (!url.startsWith('blob:')) {
      image.crossOrigin = 'Anonymous';
    }

    image.onerror = (err) => {
      console.error('Image load error:', url);
      reject(err);
    };

    image.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = image.width;
      canvas.height = image.height;

      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject(new Error('Could not get 2d context'));
        return;
      }

      ctx.drawImage(image, 0, 0);
      const imageData = ctx.getImageData(0, 0, image.width, image.height);
      const pixelData = new Uint8Array(image.width * image.height * 3);

      for (let sourceIndex = 0, pixelIndex = 0; sourceIndex < imageData.data.length; sourceIndex += 4, pixelIndex += 3) {
        pixelData[pixelIndex] = imageData.data[sourceIndex];
        pixelData[pixelIndex + 1] = imageData.data[sourceIndex + 1];
        pixelData[pixelIndex + 2] = imageData.data[sourceIndex + 2];
      }

      const imageFrame = {
        samplesPerPixel: 3,
        photometricInterpretation: 'RGB',
        planarConfiguration: 0,
        rows: image.height,
        columns: image.width,
        bitsAllocated: 8,
        bitsStored: 8,
        highBit: 7,
        pixelRepresentation: 0,
        smallestPixelValue: 0,
        largestPixelValue: 255,
        pixelData,
        pixelDataLength: pixelData.length,
      } as unknown as cornerstone.Types.IImageFrame;

      const voxelManager = cornerstone.utilities.VoxelManager.createImageVoxelManager({
        scalarData: pixelData,
        width: image.width,
        height: image.height,
        numberOfComponents: 3,
      });

      resolve({
        imageId,
        minPixelValue: 0,
        maxPixelValue: 255,
        slope: 1,
        intercept: 0,
        windowCenter: 128,
        windowWidth: 256,
        getPixelData: () => pixelData,
        rows: image.height,
        columns: image.width,
        height: image.height,
        width: image.width,
        color: true,
        rgba: false,
        columnPixelSpacing: 1,
        rowPixelSpacing: 1,
        invert: false,
        sizeInBytes: pixelData.length,
        voiLUTFunction: cornerstone.Enums.VOILUTFunctionType.LINEAR,
        getCanvas: () => canvas,
        numberOfComponents: 3,
        photometricInterpretation: 'RGB',
        imageFrame,
        voxelManager,
        dataType: 'Uint8Array',
      });
    };

    image.src = url;
  });

  return {
    promise,
    cancelFn: () => {
      console.log('Canceled image load', imageId);
    },
  };
};

const createLocalDicomImage = async (imageId: string): Promise<cornerstone.Types.IImage> => {
  const file = localDicomFiles.get(imageId);
  if (!file) {
    throw new Error(`Local DICOM not registered: ${imageId}`);
  }

  const dataSet = dicomParser.parseDicom(new Uint8Array(await file.arrayBuffer()));
  const transferSyntaxUid = dataSet.string('x00020010') || '1.2.840.10008.1.2';

  if (!['1.2.840.10008.1.2', '1.2.840.10008.1.2.1'].includes(transferSyntaxUid)) {
    throw new Error(`Unsupported local DICOM transfer syntax: ${transferSyntaxUid}`);
  }

  const {
    rows,
    columns,
    samplesPerPixel,
    photometricInterpretation,
    bitsAllocated,
    bitsStored,
    highBit,
    pixelRepresentation,
    modality,
    rowPixelSpacing,
    columnPixelSpacing,
  } = parseDicomMetadataFromDataSet(dataSet, file.name);
  const pixelElement = dataSet.elements.x7fe00010;

  if (!rows || !columns || !pixelElement) {
    throw new Error('DICOM pixel data is missing required tags.');
  }

  if (samplesPerPixel !== 1) {
    throw new Error(`Unsupported local DICOM samplesPerPixel: ${samplesPerPixel}`);
  }

  const bytesPerPixel = Math.max(1, bitsAllocated / 8);
  const singleFrameBytes = rows * columns * samplesPerPixel * bytesPerPixel;
  const pixelDataLength = pixelElement.length;

  // Detect if multi-frame and extract a single frame (middle frame)
  let frameOffset = pixelElement.dataOffset;
  if (pixelDataLength > singleFrameBytes) {
    const numberOfFrames = Math.floor(pixelDataLength / singleFrameBytes);
    const midFrame = Math.floor(numberOfFrames / 2);
    frameOffset += midFrame * singleFrameBytes;
    console.log(`[LocalDicomLoader] Multi-frame detected (${numberOfFrames} frames). Using frame ${midFrame}.`);
  }

  const pixelBytes = dataSet.byteArray.slice(frameOffset, frameOffset + singleFrameBytes);

  let scalarData: Uint8Array | Int8Array | Uint16Array | Int16Array;
  if (bitsAllocated === 8) {
    scalarData = pixelRepresentation === 1
      ? new Int8Array(pixelBytes.buffer, pixelBytes.byteOffset, pixelBytes.byteLength)
      : pixelBytes;
  } else if (bitsAllocated === 16) {
    scalarData = pixelRepresentation === 1
      ? new Int16Array(pixelBytes.buffer, pixelBytes.byteOffset, pixelBytes.byteLength / 2)
      : new Uint16Array(pixelBytes.buffer, pixelBytes.byteOffset, pixelBytes.byteLength / 2);
  } else {
    throw new Error(`Unsupported local DICOM bitsAllocated: ${bitsAllocated}`);
  }

  const { minPixelValue, maxPixelValue } = getMinMax(scalarData);
  const derivedWindowWidth = Math.max(1, maxPixelValue - minPixelValue);
  const derivedWindowCenter = minPixelValue + derivedWindowWidth / 2;
  const windowCenter = parseDicomNumber(dataSet.string('x00281050'), derivedWindowCenter);
  const windowWidth = parseDicomNumber(dataSet.string('x00281051'), derivedWindowWidth);

  localDicomMetadata.set(imageId, {
    ...parseDicomMetadataFromDataSet(dataSet, file.name),
    windowCenter,
    windowWidth,
  });

  const imageFrame = {
    samplesPerPixel,
    photometricInterpretation,
    planarConfiguration: 0,
    rows,
    columns,
    bitsAllocated,
    bitsStored,
    highBit,
    pixelRepresentation,
    pixelData: scalarData,
    pixelDataLength: scalarData.byteLength,
  } as unknown as cornerstone.Types.IImageFrame;

  const voxelManager = cornerstone.utilities.VoxelManager.createImageVoxelManager({
    scalarData,
    width: columns,
    height: rows,
    numberOfComponents: 1,
  });

  const getCanvas = () => {
    const canvas = document.createElement('canvas');
    canvas.width = columns;
    canvas.height = rows;

    const context = canvas.getContext('2d');
    if (!context) {
      return canvas;
    }

    const imageData = context.createImageData(columns, rows);
    const range = Math.max(1, maxPixelValue - minPixelValue);

    for (let pixelIndex = 0; pixelIndex < scalarData.length; pixelIndex += 1) {
      const normalized = Math.round(((Number(scalarData[pixelIndex]) - minPixelValue) / range) * 255);
      const grayscale = photometricInterpretation === 'MONOCHROME1' ? 255 - normalized : normalized;
      const rgbaOffset = pixelIndex * 4;

      imageData.data[rgbaOffset] = grayscale;
      imageData.data[rgbaOffset + 1] = grayscale;
      imageData.data[rgbaOffset + 2] = grayscale;
      imageData.data[rgbaOffset + 3] = 255;
    }

    context.putImageData(imageData, 0, 0);
    return canvas;
  };

  return {
    imageId,
    minPixelValue,
    maxPixelValue,
    slope: 1,
    intercept: 0,
    windowCenter,
    windowWidth,
    getPixelData: () => scalarData,
    rows,
    columns,
    height: rows,
    width: columns,
    color: false,
    rgba: false,
    columnPixelSpacing,
    rowPixelSpacing,
    invert: photometricInterpretation === 'MONOCHROME1',
    sizeInBytes: scalarData.byteLength,
    voiLUTFunction: cornerstone.Enums.VOILUTFunctionType.LINEAR,
    getCanvas,
    numberOfComponents: 1,
    photometricInterpretation,
    imageFrame,
    voxelManager,
    dataType: scalarData.constructor.name as any,
  };
};

const localDicomImageLoader = (imageId: string): cornerstone.Types.IImageLoadObject => ({
  promise: createLocalDicomImage(imageId),
  cancelFn: () => {
    console.log('Canceled local DICOM load', imageId);
  },
});

export function initCornerstone(): Promise<void> {
  // If already initializing or initialized, return the same promise
  if (initPromise) {
    return initPromise;
  }

  initPromise = (async () => {
    await cornerstone.init();
    await cornerstoneTools.init();

    cornerstoneDICOMImageLoader.init({
      maxWebWorkers: navigator.hardwareConcurrency || 1,
    });

    cornerstone.imageLoader.registerImageLoader('web', webImageLoader);
    cornerstone.imageLoader.registerImageLoader('dicomlocal', localDicomImageLoader);
    cornerstoneDICOMImageLoader.wadouri.register();
    cornerstoneDICOMImageLoader.wadors.register();

    // Register custom multiframe loader for DICOM files with embedded volume data
    registerMultiframeImageLoader();

    // CS3D v4 internal load volume is natively registered. No need to overwrite the unknownVolumeLoader!

    cornerstone.metaData.addProvider((type: string, imageId: string) => {
      const localDicom = localDicomMetadata.get(imageId);
      if (localDicom) {
        if (type === 'imagePixelModule') {
          return {
            pixelRepresentation: localDicom.pixelRepresentation,
            bitsAllocated: localDicom.bitsAllocated,
            bitsStored: localDicom.bitsStored,
            highBit: localDicom.highBit,
            samplesPerPixel: localDicom.samplesPerPixel,
            photometricInterpretation: localDicom.photometricInterpretation,
            planarConfiguration: 0,
          };
        }

        if (type === 'imagePlaneModule') {
          const imageOrientationPatient = localDicom.imageOrientationPatient?.length === 6
            ? localDicom.imageOrientationPatient
            : [1, 0, 0, 0, 1, 0];
          return {
            rows: localDicom.rows,
            columns: localDicom.columns,
            rowCosines: imageOrientationPatient.slice(0, 3),
            columnCosines: imageOrientationPatient.slice(3, 6),
            imageOrientationPatient,
            imagePositionPatient: localDicom.imagePositionPatient,
            frameOfReferenceUID: localDicom.frameOfReferenceUID,
            pixelSpacing: localDicom.pixelSpacing,
            columnPixelSpacing: localDicom.columnPixelSpacing,
            rowPixelSpacing: localDicom.rowPixelSpacing,
            sliceThickness: localDicom.sliceThickness,
            spacingBetweenSlices: localDicom.spacingBetweenSlices,
            sliceLocation: localDicom.imagePositionPatient[2] || 0,
          };
        }

        if (type === 'voiLutModule') {
          return {
            windowWidth: [localDicom.windowWidth],
            windowCenter: [localDicom.windowCenter],
          };
        }

        if (type === 'generalSeriesModule') {
          return {
            modality: localDicom.modality,
            seriesInstanceUID: localDicom.seriesInstanceUID,
            seriesDescription: localDicom.seriesDescription,
          };
        }

        if (type === 'generalPatientModule') {
          return {
            patientName: localDicom.patientName,
            patientId: localDicom.patientId,
            patientSex: localDicom.patientSex,
            patientBirthDate: localDicom.patientBirthDate,
          };
        }

        if (type === 'generalStudyModule') {
          return {
            studyDate: localDicom.studyDate,
            studyTime: localDicom.studyTime,
            accessionNumber: localDicom.accessionNumber,
            studyDescription: localDicom.studyDescription,
            institutionName: localDicom.institutionName,
          };
        }

        if (type === 'generalImageModule') {
          return {
            instanceNumber: localDicom.instanceNumber,
            imageType: localDicom.imageType,
          };
        }

        if (type === 'transferSyntax') {
          return {
            transferSyntaxUID: localDicom.transferSyntaxUid,
          };
        }

        return undefined;
      }

      const isWeb = imageId.startsWith('web:');

      if (!isWeb) {
        return undefined;
      }

      if (type === 'imagePixelModule' && isWeb) {
        return {
          pixelRepresentation: 0,
          bitsAllocated: 8,
          bitsStored: 8,
          highBit: 7,
          samplesPerPixel: 3,
          photometricInterpretation: 'RGB',
          planarConfiguration: 0,
        };
      }

      if (type === 'imagePlaneModule') {
        return {
          columnCosines: [1, 0, 0],
          rowCosines: [0, 1, 0],
          imageOrientationPatient: [1, 0, 0, 0, 1, 0],
          imagePositionPatient: [0, 0, 0],
          columnPixelSpacing: 1,
          rowPixelSpacing: 1,
          sliceThickness: 1,
          sliceLocation: 0,
        };
      }

      if (type === 'voiLutModule') {
        return {
          windowWidth: [256],
          windowCenter: [128],
        };
      }

      if (type === 'generalSeriesModule') {
        return {
          modality: 'OT',
        };
      }

      return undefined;
    });

    console.log('Cornerstone initialized');
  })();

  // If initialization fails, allow retry by clearing the promise
  initPromise.catch(() => {
    initPromise = null;
  });

  return initPromise;
}

