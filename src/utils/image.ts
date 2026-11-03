import { SAMPLE_MIME, SAMPLE_QUALITY } from '../config';

export async function loadImage(src: string): Promise<HTMLImageElement> {
  const img = new Image();
  img.decoding = 'async';
  img.src = src;
  await img.decode();
  return img;
}

/** Decodes an uploaded file (respecting EXIF orientation where the browser does). */
export async function decodeFile(file: Blob): Promise<ImageBitmap | HTMLImageElement> {
  try {
    return await createImageBitmap(file);
  } catch {
    const url = URL.createObjectURL(file);
    try {
      return await loadImage(url);
    } finally {
      URL.revokeObjectURL(url);
    }
  }
}

export function closeImage(image: ImageBitmap | HTMLImageElement): void {
  if ('close' in image) image.close();
}

/** Face crops are stored as 224x224 JPEG data URLs (thumbnail and model input in one). */
export function canvasToSample(canvas: HTMLCanvasElement): string {
  return canvas.toDataURL(SAMPLE_MIME, SAMPLE_QUALITY);
}

let snapshotCanvas: HTMLCanvasElement | null = null;

/**
 * Copies the current video frame to a canvas so detection and cropping work on
 * the exact same pixels. Not mirrored: samples keep the true orientation, like
 * uploaded photos (the horizontal-flip augmentation covers both anyway).
 */
export function snapshotVideo(video: HTMLVideoElement, canvas?: HTMLCanvasElement): HTMLCanvasElement {
  const c = canvas ?? (snapshotCanvas ??= document.createElement('canvas'));
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext('2d')!.drawImage(video, 0, 0);
  return c;
}
