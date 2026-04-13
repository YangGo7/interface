/**
 * Custom multi-frame image loader for DICOM files that contain multiple slices
 * in a single file but lack the standard NumberOfFrames (0028,0008) tag.
 *
 * Cornerstone's built-in wadouri multi-frame support requires the NumberOfFrames
 * tag. This loader manually splits the pixel data buffer into per-frame images.
 *
 * Scheme: "multiframe"
 * ImageId format: "multiframe:<registrationKey>&frame=<0-based frameIndex>"
 */

import * as cornerstone from '@cornerstonejs/core';
import dicomParser from 'dicom-parser';
import { DicomOverlayMetadata, parseDicomMetadataFromDataSet } from './dicomMetadata';

export interface MultiframeDicomRegistration {
    key: string;
    file: File;
    dataSet: dicomParser.DataSet;
    metadata: DicomOverlayMetadata;
    rows: number;
    columns: number;
    bitsAllocated: number;
    pixelRepresentation: number;
    samplesPerPixel: number;
    photometricInterpretation: string;
    bytesPerPixel: number;
    singleFrameBytes: number;
    numberOfFrames: number;
    transferSyntaxUid: string;
    rescaleSlope: number;
    rescaleIntercept: number;
    windowCenter: number;
    windowWidth: number;
    defaultWindowCenter: number;
    defaultWindowWidth: number;
    pixelSpacing: number[];
    imageOrientationPatient: number[];
    imagePositionPatient: number[];
    sliceThickness: number;
    frameOfReferenceUID: string;
    modality: string;
    seriesInstanceUID: string;
    perFramePixelSpacing: number[][];
    perFrameImageOrientationPatient: number[][];
    perFrameImagePositionPatient: number[][];
    perFrameSliceThickness: number[];
}

const registrations = new Map<string, MultiframeDicomRegistration>();

const parseNum = (v: string | undefined, def: number) => {
    if (!v) return def;
    const n = Number(v.split('\\')[0]);
    return Number.isFinite(n) ? n : def;
};

const parseNumList = (v: string | undefined, def: number[]) => {
    if (!v) return def;
    const nums = v.split('\\').map(Number).filter(Number.isFinite);
    return nums.length > 0 ? nums : def;
};

const getSequenceItemDataSet = (dataSet: dicomParser.DataSet | undefined, sequenceTag: string, itemIndex = 0) => {
    const sequence = dataSet?.elements?.[sequenceTag as keyof typeof dataSet.elements] as any;
    const item = sequence?.items?.[itemIndex];
    return item?.dataSet as dicomParser.DataSet | undefined;
};

const computeFallbackFramePosition = (
    imageOrientationPatient: number[],
    imagePositionPatient: number[],
    sliceThickness: number,
    frame: number
) => {
    const scanAxisNormal = [
        imageOrientationPatient[1] * imageOrientationPatient[5] - imageOrientationPatient[2] * imageOrientationPatient[4],
        imageOrientationPatient[2] * imageOrientationPatient[3] - imageOrientationPatient[0] * imageOrientationPatient[5],
        imageOrientationPatient[0] * imageOrientationPatient[4] - imageOrientationPatient[1] * imageOrientationPatient[3],
    ];

    return [
        imagePositionPatient[0] + scanAxisNormal[0] * sliceThickness * frame,
        imagePositionPatient[1] + scanAxisNormal[1] * sliceThickness * frame,
        imagePositionPatient[2] + scanAxisNormal[2] * sliceThickness * frame,
    ];
};

const buildPerFrameGeometry = (
    dataSet: dicomParser.DataSet,
    numberOfFrames: number,
    pixelSpacing: number[],
    imageOrientationPatient: number[],
    imagePositionPatient: number[],
    sliceThickness: number
) => {
    const sharedFunctionalGroups = getSequenceItemDataSet(dataSet, 'x52009229');
    const perFrameSequence = (dataSet.elements.x52009230 as any)?.items || [];

    const sharedPixelMeasures = getSequenceItemDataSet(sharedFunctionalGroups, 'x00289110');
    const sharedPlaneOrientation = getSequenceItemDataSet(sharedFunctionalGroups, 'x00209116');
    const sharedPlanePosition = getSequenceItemDataSet(sharedFunctionalGroups, 'x00209113');

    const sharedPixelSpacing = parseNumList(sharedPixelMeasures?.string('x00280030'), pixelSpacing);
    const sharedSliceThickness = parseNum(
        sharedPixelMeasures?.string('x00180088') || sharedPixelMeasures?.string('x00180050'),
        sliceThickness
    );
    const sharedImageOrientationPatient = parseNumList(
        sharedPlaneOrientation?.string('x00200037'),
        imageOrientationPatient
    );
    const sharedImagePositionPatient = parseNumList(
        sharedPlanePosition?.string('x00200032'),
        imagePositionPatient
    );

    const perFramePixelSpacing: number[][] = [];
    const perFrameImageOrientationPatient: number[][] = [];
    const perFrameImagePositionPatient: number[][] = [];
    const perFrameSliceThickness: number[] = [];

    for (let frame = 0; frame < numberOfFrames; frame += 1) {
        const frameDataSet = perFrameSequence[frame]?.dataSet as dicomParser.DataSet | undefined;
        const framePixelMeasures = getSequenceItemDataSet(frameDataSet, 'x00289110');
        const framePlaneOrientation = getSequenceItemDataSet(frameDataSet, 'x00209116');
        const framePlanePosition = getSequenceItemDataSet(frameDataSet, 'x00209113');

        const framePixelSpacing = parseNumList(
            framePixelMeasures?.string('x00280030'),
            sharedPixelSpacing
        );
        const frameSliceThickness = parseNum(
            framePixelMeasures?.string('x00180088') || framePixelMeasures?.string('x00180050'),
            sharedSliceThickness
        );
        const frameImageOrientationPatient = parseNumList(
            framePlaneOrientation?.string('x00200037'),
            sharedImageOrientationPatient
        );
        const explicitFramePosition = parseNumList(
            framePlanePosition?.string('x00200032'),
            []
        );
        const frameImagePositionPatient = explicitFramePosition.length === 3
            ? explicitFramePosition
            : computeFallbackFramePosition(
                frameImageOrientationPatient,
                sharedImagePositionPatient,
                frameSliceThickness,
                frame
            );

        perFramePixelSpacing.push(framePixelSpacing);
        perFrameSliceThickness.push(frameSliceThickness);
        perFrameImageOrientationPatient.push(frameImageOrientationPatient);
        perFrameImagePositionPatient.push(frameImagePositionPatient);
    }

    return {
        perFramePixelSpacing,
        perFrameImageOrientationPatient,
        perFrameImagePositionPatient,
        perFrameSliceThickness,
    };
};

const roundForLog = (value: number) => Math.round(value * 1000) / 1000;

const summarizeFrameGeometryForLog = (registration: MultiframeDicomRegistration) => {
    const sampleCount = Math.min(registration.numberOfFrames, 3);
    const firstFrames = Array.from({ length: sampleCount }, (_, frame) => ({
        frame,
        ipp: (registration.perFrameImagePositionPatient[frame] || registration.imagePositionPatient).map(roundForLog),
        iop: (registration.perFrameImageOrientationPatient[frame] || registration.imageOrientationPatient).map(roundForLog),
        spacing: (registration.perFramePixelSpacing[frame] || registration.pixelSpacing).map(roundForLog),
        thickness: roundForLog(registration.perFrameSliceThickness[frame] || registration.sliceThickness),
    }));
    const lastFrame = Math.max(0, registration.numberOfFrames - 1);

    return {
        frames: registration.numberOfFrames,
        dicomWindowCenter: roundForLog(registration.windowCenter),
        dicomWindowWidth: roundForLog(registration.windowWidth),
        defaultWindowCenter: roundForLog(registration.defaultWindowCenter),
        defaultWindowWidth: roundForLog(registration.defaultWindowWidth),
        sharedSpacing: registration.pixelSpacing.map(roundForLog),
        sharedThickness: roundForLog(registration.sliceThickness),
        sharedIOP: registration.imageOrientationPatient.map(roundForLog),
        firstFrames,
        lastFrame: {
            frame: lastFrame,
            ipp: (registration.perFrameImagePositionPatient[lastFrame] || registration.imagePositionPatient).map(roundForLog),
        },
    };
};

const extractFrameScalarData = (
    dataSet: dicomParser.DataSet,
    frameOffset: number,
    singleFrameBytes: number,
    bitsAllocated: number,
    pixelRepresentation: number
) => {
    if (bitsAllocated === 16) {
        const rawBytes = dataSet.byteArray.buffer.slice(frameOffset, frameOffset + singleFrameBytes);
        return pixelRepresentation === 1
            ? new Int16Array(rawBytes)
            : new Uint16Array(rawBytes);
    }

    if (bitsAllocated === 8) {
        return new Uint8Array(dataSet.byteArray.buffer.slice(frameOffset, frameOffset + singleFrameBytes));
    }

    if (bitsAllocated === 32) {
        const rawBytes = dataSet.byteArray.buffer.slice(frameOffset, frameOffset + singleFrameBytes);
        return new Float32Array(rawBytes);
    }

    throw new Error(`Unsupported bitsAllocated: ${bitsAllocated}`);
};

const computeRobustWindowFromFrame = (
    dataSet: dicomParser.DataSet,
    frameOffset: number,
    singleFrameBytes: number,
    bitsAllocated: number,
    pixelRepresentation: number,
    slope: number,
    intercept: number
) => {
    const scalarData = extractFrameScalarData(
        dataSet,
        frameOffset,
        singleFrameBytes,
        bitsAllocated,
        pixelRepresentation
    );

    const values = new Float64Array(scalarData.length);
    for (let index = 0; index < scalarData.length; index += 1) {
        values[index] = Number(scalarData[index]) * slope + intercept;
    }

    const sorted = Array.from(values).sort((a, b) => a - b);
    const percentile = (ratio: number) => {
        const clamped = Math.min(1, Math.max(0, ratio));
        const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(clamped * (sorted.length - 1))));
        return sorted[idx];
    };

    const low = percentile(0.05);
    const high = percentile(0.995);
    const width = Math.max(1, high - low);
    const center = low + width / 2;

    return {
        defaultWindowCenter: center,
        defaultWindowWidth: width,
    };
};

/**
 * Parse a raw DICOM file, detect multi-frame layout, and register for use with
 * the custom "multiframe" image loader.
 *
 * Returns the registration info including numberOfFrames.
 */
export async function registerMultiframeDicomFile(
    file: File
): Promise<MultiframeDicomRegistration> {
    const key = `mf_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const arrayBuffer = await file.arrayBuffer();
    const byteArray = new Uint8Array(arrayBuffer);
    const dataSet = dicomParser.parseDicom(byteArray);
    const baseMetadata = parseDicomMetadataFromDataSet(dataSet, file.name);

    const rows = dataSet.uint16('x00280010') || 0;
    const columns = dataSet.uint16('x00280011') || 0;
    const bitsAllocated = dataSet.uint16('x00280100') || 16;
    const pixelRepresentation = dataSet.uint16('x00280103') || 0;
    const samplesPerPixel = dataSet.uint16('x00280002') || 1;
    const photometricInterpretation = dataSet.string('x00280004') || 'MONOCHROME2';
    const transferSyntaxUid = dataSet.string('x00020010') || '1.2.840.10008.1.2';

    const bytesPerPixel = Math.max(1, bitsAllocated / 8);
    const singleFrameBytes = rows * columns * samplesPerPixel * bytesPerPixel;

    // Detect number of frames
    const pixelElement = dataSet.elements.x7fe00010;
    const pixelDataLength = pixelElement?.length || 0;
    let numberOfFrames = 1;
    const nfTag = dataSet.string('x00280008');
    if (nfTag) {
        numberOfFrames = parseInt(nfTag, 10) || 1;
    } else if (singleFrameBytes > 0 && pixelDataLength > singleFrameBytes) {
        numberOfFrames = Math.floor(pixelDataLength / singleFrameBytes);
    }

    // Parse spacing and orientation
    const pixelSpacing = parseNumList(dataSet.string('x00280030'), [1, 1]);
    const imageOrientationPatient = parseNumList(dataSet.string('x00200037'), [1, 0, 0, 0, 1, 0]);
    const imagePositionPatient = parseNumList(dataSet.string('x00200032'), [0, 0, 0]);
    // Try Spacing Between Slices (0018,0088) first, then Slice Thickness (0018,0050)
    const spacingTag = dataSet.string('x00180088') || dataSet.string('x00180050');
    const sliceThickness = parseNum(spacingTag, pixelSpacing[0] || 1);
    const rescaleSlope = parseNum(dataSet.string('x00281053'), 1);
    const rescaleIntercept = parseNum(dataSet.string('x00281052'), 0);
    const windowCenter = parseNum(dataSet.string('x00281050'), 0);
    const windowWidth = parseNum(dataSet.string('x00281051'), 0);
    const middleFrame = Math.max(0, Math.floor(numberOfFrames / 2));
    const middleFrameOffset = (pixelElement?.dataOffset || 0) + middleFrame * singleFrameBytes;
    const { defaultWindowCenter, defaultWindowWidth } = computeRobustWindowFromFrame(
        dataSet,
        middleFrameOffset,
        singleFrameBytes,
        bitsAllocated,
        pixelRepresentation,
        rescaleSlope,
        rescaleIntercept
    );
    const frameOfReferenceUID = dataSet.string('x00200052') || `generated.${key}`;
    const modality = dataSet.string('x00080060') || 'CT';
    const seriesInstanceUID = dataSet.string('x0020000e') || `generated.series.${key}`;
    const {
        perFramePixelSpacing,
        perFrameImageOrientationPatient,
        perFrameImagePositionPatient,
        perFrameSliceThickness,
    } = buildPerFrameGeometry(
        dataSet,
        numberOfFrames,
        pixelSpacing,
        imageOrientationPatient,
        imagePositionPatient,
        sliceThickness
    );

    const registration: MultiframeDicomRegistration = {
        key,
        file,
        dataSet,
        metadata: {
            ...baseMetadata,
            numberOfFrames,
            windowCenter: windowCenter || baseMetadata.windowCenter,
            windowWidth: windowWidth || baseMetadata.windowWidth,
            sliceThickness,
            spacingBetweenSlices: sliceThickness,
            pixelSpacing: [pixelSpacing[0] || 1, pixelSpacing[1] || 1],
            rowPixelSpacing: pixelSpacing[0] || 1,
            columnPixelSpacing: pixelSpacing[1] || 1,
            imageOrientationPatient,
            imagePositionPatient,
            modality,
            seriesInstanceUID,
            frameOfReferenceUID,
            transferSyntaxUid,
        },
        rows,
        columns,
        bitsAllocated,
        pixelRepresentation,
        samplesPerPixel,
        photometricInterpretation,
        bytesPerPixel,
        singleFrameBytes,
        numberOfFrames,
        transferSyntaxUid,
        rescaleSlope,
        rescaleIntercept,
        windowCenter,
        windowWidth,
        defaultWindowCenter,
        defaultWindowWidth,
        pixelSpacing,
        imageOrientationPatient,
        imagePositionPatient,
        sliceThickness,
        frameOfReferenceUID,
        modality,
        seriesInstanceUID,
        perFramePixelSpacing,
        perFrameImageOrientationPatient,
        perFrameImagePositionPatient,
        perFrameSliceThickness,
    };

    registrations.set(key, registration);
    console.log('[MultiframeLoader] Geometry summary', summarizeFrameGeometryForLog(registration));

    console.log(`[MultiframeLoader] Registered "${file.name}": ${numberOfFrames} frames, ${rows}×${columns}, ${bitsAllocated}-bit`);
    return registration;
}

/**
 * Generate imageIds for all frames in a registered multi-frame file.
 */
export function generateMultiframeImageIds(reg: MultiframeDicomRegistration): string[] {
    const ids: string[] = [];
    for (let f = 0; f < reg.numberOfFrames; f++) {
        ids.push(`multiframe:${reg.key}&frame=${f}`);
    }
    return ids;
}

export function getMultiframeDicomMetadata(imageId?: string | null) {
    if (!imageId || !imageId.startsWith('multiframe:')) {
        return null;
    }

    const withoutScheme = imageId.replace('multiframe:', '');
    const frameIdx = withoutScheme.indexOf('&frame=');
    const key = frameIdx > -1 ? withoutScheme.substring(0, frameIdx) : withoutScheme;
    return registrations.get(key)?.metadata || null;
}

/**
 * The custom image loader function, to be registered with cornerstone.
 */
function multiframeImageLoader(
    imageId: string,
    options?: Record<string, any>
): cornerstone.Types.IImageLoadObject {
    const promise = new Promise<cornerstone.Types.IImage>((resolve, reject) => {
        try {
            // Parse imageId: "multiframe:<key>&frame=<N>"
            const withoutScheme = imageId.replace('multiframe:', '');
            const frameIdx = withoutScheme.indexOf('&frame=');
            const key = frameIdx > -1 ? withoutScheme.substring(0, frameIdx) : withoutScheme;
            const frame = frameIdx > -1 ? parseInt(withoutScheme.substring(frameIdx + 7), 10) : 0;

            const reg = registrations.get(key);
            if (!reg) {
                throw new Error(`Multiframe registration not found for key: ${key}`);
            }

            const { dataSet, rows, columns, bitsAllocated, pixelRepresentation, samplesPerPixel,
                singleFrameBytes, bytesPerPixel, rescaleSlope, rescaleIntercept,
                windowCenter, windowWidth, defaultWindowCenter, defaultWindowWidth, photometricInterpretation,
                perFrameSliceThickness, perFramePixelSpacing,
                perFrameImageOrientationPatient, perFrameImagePositionPatient } = reg;

            // Extract pixel data for this frame
            const pixelElement = dataSet.elements.x7fe00010;
            if (!pixelElement) {
                throw new Error('No pixel data element found');
            }

            const pixelDataOffset = pixelElement.dataOffset;
            const frameOffset = pixelDataOffset + frame * singleFrameBytes;

            let pixelData: Int16Array | Uint16Array | Uint8Array | Float32Array;

            if (bitsAllocated === 16) {
                const rawBytes = dataSet.byteArray.buffer.slice(frameOffset, frameOffset + singleFrameBytes);
                pixelData = pixelRepresentation === 1
                    ? new Int16Array(rawBytes)
                    : new Uint16Array(rawBytes);
            } else if (bitsAllocated === 8) {
                pixelData = new Uint8Array(dataSet.byteArray.buffer, frameOffset, singleFrameBytes);
            } else if (bitsAllocated === 32) {
                const rawBytes = dataSet.byteArray.buffer.slice(frameOffset, frameOffset + singleFrameBytes);
                pixelData = new Float32Array(rawBytes);
            } else {
                throw new Error(`Unsupported bitsAllocated: ${bitsAllocated}`);
            }

            // Compute min/max
            let minPixelValue = Infinity;
            let maxPixelValue = -Infinity;
            for (let i = 0; i < pixelData.length; i++) {
                if (pixelData[i] < minPixelValue) minPixelValue = pixelData[i];
                if (pixelData[i] > maxPixelValue) maxPixelValue = pixelData[i];
            }

            const framePixelSpacing = perFramePixelSpacing[frame] || reg.pixelSpacing;
            const frameImageOrientationPatient = perFrameImageOrientationPatient[frame] || reg.imageOrientationPatient;
            const frameImagePositionPatient = perFrameImagePositionPatient[frame] || reg.imagePositionPatient;
            const frameSliceThickness = perFrameSliceThickness[frame] || reg.sliceThickness;

            const image: cornerstone.Types.IImage = {
                imageId,
                minPixelValue,
                maxPixelValue,
                slope: rescaleSlope,
                intercept: rescaleIntercept,
                windowCenter: defaultWindowCenter,
                windowWidth: defaultWindowWidth,
                getPixelData: () => pixelData,
                getCanvas: undefined,
                rows,
                columns,
                height: rows,
                width: columns,
                color: samplesPerPixel > 1,
                rgba: false,
                numComps: samplesPerPixel,
                columnPixelSpacing: framePixelSpacing[1],
                rowPixelSpacing: framePixelSpacing[0],
                sliceThickness: frameSliceThickness,
                invert: photometricInterpretation === 'MONOCHROME1',
                sizeInBytes: singleFrameBytes,
                dataType: pixelRepresentation === 1 ? 'Int16Array' : (bitsAllocated === 8 ? 'Uint8Array' : 'Uint16Array'),
                imageQualityStatus: 2, // FULL_RESOLUTION
            } as any;

            resolve(image);
        } catch (err) {
            reject(err);
        }
    });

    return { promise, cancelFn: undefined };
}

/**
 * Register the multiframe metadata provider with cornerstone.
 * Call this during initialization.
 */
export function registerMultiframeMetadataProvider() {
    cornerstone.metaData.addProvider((type: string, imageId: string) => {
        if (!imageId.startsWith('multiframe:')) return undefined;

        const withoutScheme = imageId.replace('multiframe:', '');
        const frameIdx = withoutScheme.indexOf('&frame=');
        const key = frameIdx > -1 ? withoutScheme.substring(0, frameIdx) : withoutScheme;
        const frame = frameIdx > -1 ? parseInt(withoutScheme.substring(frameIdx + 7), 10) : 0;

        const reg = registrations.get(key);
        if (!reg) return undefined;

        const frameIPP = reg.perFrameImagePositionPatient[frame] || reg.imagePositionPatient;
        const frameIOP = reg.perFrameImageOrientationPatient[frame] || reg.imageOrientationPatient;
        const framePixelSpacing = reg.perFramePixelSpacing[frame] || reg.pixelSpacing;
        const frameSliceThickness = reg.perFrameSliceThickness[frame] || reg.sliceThickness;
        const rowCosines = frameIOP.slice(0, 3);
        const columnCosines = frameIOP.slice(3, 6);
        const scanAxisNormal = [
            rowCosines[1] * columnCosines[2] - rowCosines[2] * columnCosines[1],
            rowCosines[2] * columnCosines[0] - rowCosines[0] * columnCosines[2],
            rowCosines[0] * columnCosines[1] - rowCosines[1] * columnCosines[0],
        ];
        const sliceLocation =
            frameIPP[0] * scanAxisNormal[0] +
            frameIPP[1] * scanAxisNormal[1] +
            frameIPP[2] * scanAxisNormal[2];

        if (type === 'imagePixelModule') {
            return {
                pixelRepresentation: reg.pixelRepresentation,
                bitsAllocated: reg.bitsAllocated,
                bitsStored: reg.bitsAllocated, // assume bitsStored = bitsAllocated
                highBit: reg.bitsAllocated - 1,
                samplesPerPixel: reg.samplesPerPixel,
                photometricInterpretation: reg.photometricInterpretation,
                planarConfiguration: 0,
            };
        }

        if (type === 'imagePlaneModule') {
            return {
                frameOfReferenceUID: reg.frameOfReferenceUID,
                rows: reg.rows,
                columns: reg.columns,
                imageOrientationPatient: frameIOP,
                rowCosines,
                columnCosines,
                imagePositionPatient: frameIPP,
                sliceThickness: frameSliceThickness,
                sliceLocation,
                pixelSpacing: framePixelSpacing,
                rowPixelSpacing: framePixelSpacing[0],
                columnPixelSpacing: framePixelSpacing[1],
            };
        }

        if (type === 'voiLutModule') {
            return {
                windowWidth: [reg.defaultWindowWidth],
                windowCenter: [reg.defaultWindowCenter],
            };
        }

        if (type === 'modalityLutModule') {
            return {
                rescaleIntercept: reg.rescaleIntercept,
                rescaleSlope: reg.rescaleSlope,
            };
        }

        if (type === 'generalSeriesModule') {
            return {
                modality: reg.modality,
                seriesInstanceUID: reg.seriesInstanceUID,
                seriesDescription: reg.metadata.seriesDescription,
            };
        }

        if (type === 'generalPatientModule') {
            return {
                patientName: reg.metadata.patientName,
                patientId: reg.metadata.patientId,
                patientSex: reg.metadata.patientSex,
                patientBirthDate: reg.metadata.patientBirthDate,
            };
        }

        if (type === 'generalStudyModule') {
            return {
                studyDate: reg.metadata.studyDate,
                studyTime: reg.metadata.studyTime,
                accessionNumber: reg.metadata.accessionNumber,
                studyDescription: reg.metadata.studyDescription,
                institutionName: reg.metadata.institutionName,
            };
        }

        if (type === 'generalImageModule') {
            return {
                instanceNumber: reg.metadata.instanceNumber,
                imageType: reg.metadata.imageType,
            };
        }

        if (type === 'transferSyntax') {
            return {
                transferSyntaxUID: reg.transferSyntaxUid,
            };
        }

        return undefined;
    }, 10000); // high priority
}

/**
 * Register the multiframe image loader scheme with cornerstone.
 * Call this during initialization.
 */
export function registerMultiframeImageLoader() {
    cornerstone.imageLoader.registerImageLoader('multiframe', multiframeImageLoader);
    registerMultiframeMetadataProvider();
    console.log('[MultiframeLoader] Registered "multiframe" image loader and metadata provider');
}
