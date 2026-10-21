import { PICO_FACEFINDER_B64 } from './picoCascade'

export interface FaceDetection {
  row: number
  col: number
  size: number
  score: number
}

type ClassifyFn = (r: number, c: number, s: number, pixels: Uint8Array, ldim: number) => number

let cachedClassifier: ClassifyFn | null = null

function unpackCascade(bytes: Uint8Array): ClassifyFn {
  const dview = new DataView(new ArrayBuffer(4))
  let p = 8
  dview.setUint8(0, bytes[p + 0])
  dview.setUint8(1, bytes[p + 1])
  dview.setUint8(2, bytes[p + 2])
  dview.setUint8(3, bytes[p + 3])
  const tdepth = dview.getInt32(0, true)
  p += 4
  dview.setUint8(0, bytes[p + 0])
  dview.setUint8(1, bytes[p + 1])
  dview.setUint8(2, bytes[p + 2])
  dview.setUint8(3, bytes[p + 3])
  const ntrees = dview.getInt32(0, true)
  p += 4

  const tcodes: number[] = []
  const tpreds: number[] = []
  const thresh: number[] = []

  for (let t = 0; t < ntrees; ++t) {
    tcodes.push(0, 0, 0, 0)
    for (let i = 0; i < 4 * Math.pow(2, tdepth) - 4; ++i) {
      tcodes.push(bytes[p + i])
    }
    p += 4 * Math.pow(2, tdepth) - 4

    for (let i = 0; i < Math.pow(2, tdepth); ++i) {
      dview.setUint8(0, bytes[p + 0])
      dview.setUint8(1, bytes[p + 1])
      dview.setUint8(2, bytes[p + 2])
      dview.setUint8(3, bytes[p + 3])
      tpreds.push(dview.getFloat32(0, true))
      p += 4
    }

    dview.setUint8(0, bytes[p + 0])
    dview.setUint8(1, bytes[p + 1])
    dview.setUint8(2, bytes[p + 2])
    dview.setUint8(3, bytes[p + 3])
    thresh.push(dview.getFloat32(0, true))
    p += 4
  }

  const tcodesArr = new Int8Array(tcodes)
  const tpredsArr = new Float32Array(tpreds)
  const threshArr = new Float32Array(thresh)

  return function classifyRegion(r: number, c: number, s: number, pixels: Uint8Array, ldim: number): number {
    r = 256 * r
    c = 256 * c
    let root = 0
    let o = 0.0
    const pow2tdepth = Math.pow(2, tdepth) >> 0

    for (let i = 0; i < ntrees; ++i) {
      let idx = 1
      for (let j = 0; j < tdepth; ++j) {
        const p1 = ((r + tcodesArr[root + 4 * idx + 0] * s) >> 8) * ldim + ((c + tcodesArr[root + 4 * idx + 1] * s) >> 8)
        const p2 = ((r + tcodesArr[root + 4 * idx + 2] * s) >> 8) * ldim + ((c + tcodesArr[root + 4 * idx + 3] * s) >> 8)
        idx = 2 * idx + (pixels[p1] <= pixels[p2] ? 1 : 0)
      }
      o += tpredsArr[pow2tdepth * i + idx - pow2tdepth]
      if (o <= threshArr[i]) return -1
      root += 4 * pow2tdepth
    }
    return o - threshArr[ntrees - 1]
  }
}

function getClassifier(): ClassifyFn {
  if (!cachedClassifier) {
    const binaryStr = atob(PICO_FACEFINDER_B64)
    const len = binaryStr.length
    const bytes = new Uint8Array(len)
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryStr.charCodeAt(i)
    }
    cachedClassifier = unpackCascade(bytes)
  }
  return cachedClassifier
}

export function detectFacesInCanvas(canvas: HTMLCanvasElement): FaceDetection[] {
  const width = canvas.width
  const height = canvas.height
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) return []

  const imgData = ctx.getImageData(0, 0, width, height)
  const d = imgData.data
  const gray = new Uint8Array(width * height)

  for (let i = 0; i < width * height; i++) {
    // Standard perceptual luminance: 0.299R + 0.587G + 0.114B
    gray[i] = (2 * d[i * 4] + 7 * d[i * 4 + 1] + 1 * d[i * 4 + 2]) / 10
  }

  const classifier = getClassifier()
  const minsize = 28
  const maxsize = Math.min(width, height)
  const scalefactor = 1.15
  const shiftfactor = 0.1

  let scale = minsize
  const detections: FaceDetection[] = []

  while (scale <= maxsize) {
    const step = Math.max(shiftfactor * scale, 1) >> 0
    const offset = (scale / 2 + 1) >> 0

    for (let r = offset; r <= height - offset; r += step) {
      for (let c = offset; c <= width - offset; c += step) {
        const q = classifier(r, c, scale, gray, width)
        if (q > 0.0) {
          detections.push({ row: r, col: c, size: scale, score: q })
        }
      }
    }
    scale *= scalefactor
  }

  return detections
}
