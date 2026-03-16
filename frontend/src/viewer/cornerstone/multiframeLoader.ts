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

export interface MultiframeDicomRegistration {
    key: string;
    file: File;
    dataSet: dicomParser.DataSet;
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
    pixelSpacing: number[];
    imageOrientationPatient: number[];
    imagePositionPatient: number[];
    sliceThickness: number;
    frameOfReferenceUID: string;
    modality: string;
    seriesInstanceUID: string;
}

const registrations = new Map<string, MultiframeDicomRegistration>();

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
    const frameOfReferenceUID = dataSet.string('x00200052') || `generated.${key}`;
    const modality = dataSet.string('x00080060') || 'CT';
    const seriesInstanceUID = dataSet.string('x0020000e') || `generated.series.${key}`;

    const registration: MultiframeDicomRegistration = {
        key,
        file,
        dataSet,
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
        pixelSpacing,
        imageOrientationPatient,
        imagePositionPatient,
        sliceThickness,
        frameOfReferenceUID,
        modality,
        seriesInstanceUID,
    };

    registrations.set(key, registration);

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
                windowCenter, windowWidth, photometricInterpretation, sliceThickness,
                pixelSpacing, imageOrientationPatient, imagePositionPatient } = reg;

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

            // Compute image position for this frame (shift along Z axis)
            const scanAxisNormal = [
                imageOrientationPatient[1] * imageOrientationPatient[5] - imageOrientationPatient[2] * imageOrientationPatient[4],
                imageOrientationPatient[2] * imageOrientationPatient[3] - imageOrientationPatient[0] * imageOrientationPatient[5],
                imageOrientationPatient[0] * imageOrientationPatient[4] - imageOrientationPatient[1] * imageOrientationPatient[3],
            ];
            const frameImagePositionPatient = [
                imagePositionPatient[0] + scanAxisNormal[0] * sliceThickness * frame,
                imagePositionPatient[1] + scanAxisNormal[1] * sliceThickness * frame,
                imagePositionPatient[2] + scanAxisNormal[2] * sliceThickness * frame,
            ];

            const image: cornerstone.Types.IImage = {
                imageId,
                minPixelValue,
                maxPixelValue,
                slope: rescaleSlope,
                intercept: rescaleIntercept,
                windowCenter: windowCenter || (minPixelValue + maxPixelValue) / 2,
                windowWidth: windowWidth || (maxPixelValue - minPixelValue),
                getPixelData: () => pixelData,
                getCanvas: undefined,
                rows,
                columns,
                height: rows,
                width: columns,
                color: samplesPerPixel > 1,
                rgba: false,
                numComps: samplesPerPixel,
                columnPixelSpacing: pixelSpacing[1],
                rowPixelSpacing: pixelSpacing[0],
                sliceThickness,
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

        const scanAxisNormal = [
            reg.imageOrientationPatient[1] * reg.imageOrientationPatient[5] - reg.imageOrientationPatient[2] * reg.imageOrientationPatient[4],
            reg.imageOrientationPatient[2] * reg.imageOrientationPatient[3] - reg.imageOrientationPatient[0] * reg.imageOrientationPatient[5],
            reg.imageOrientationPatient[0] * reg.imageOrientationPatient[4] - reg.imageOrientationPatient[1] * reg.imageOrientationPatient[3],
        ];
        const frameIPP = [
            reg.imagePositionPatient[0] + scanAxisNormal[0] * reg.sliceThickness * frame,
            reg.imagePositionPatient[1] + scanAxisNormal[1] * reg.sliceThickness * frame,
            reg.imagePositionPatient[2] + scanAxisNormal[2] * reg.sliceThickness * frame,
        ];

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
                imageOrientationPatient: reg.imageOrientationPatient,
                rowCosines: reg.imageOrientationPatient.slice(0, 3),
                columnCosines: reg.imageOrientationPatient.slice(3, 6),
                imagePositionPatient: frameIPP,
                sliceThickness: reg.sliceThickness,
                sliceLocation: frameIPP[2],
                pixelSpacing: reg.pixelSpacing,
                rowPixelSpacing: reg.pixelSpacing[0],
                columnPixelSpacing: reg.pixelSpacing[1],
            };
        }

        if (type === 'voiLutModule') {
            return {
                windowWidth: reg.windowWidth ? [reg.windowWidth] : undefined,
                windowCenter: reg.windowCenter ? [reg.windowCenter] : undefined,
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
