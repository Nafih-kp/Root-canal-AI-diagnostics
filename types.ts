
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectionResult {
  box: BoundingBox;
  label: string;
  confidence: number;
}
