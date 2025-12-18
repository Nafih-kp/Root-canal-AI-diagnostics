import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

// Initialize TensorFlow.js backend for fallback
(async () => {
  try {
    await tf.setBackend('webgl');
    await tf.ready();
    console.log('TensorFlow.js WebGL backend initialized');
  } catch (e) {
    console.log('WebGL failed, trying WASM...', e);
    try {
      await tf.setBackend('wasm');
      await tf.ready();
      console.log('TensorFlow.js WASM backend initialized');
    } catch (e2) {
      console.log('WASM failed, using CPU...', e2);
      try {
        await tf.setBackend('cpu');
        await tf.ready();
        console.log('TensorFlow.js CPU backend initialized');
      } catch (e3) {
        console.error('All TF.js backends failed:', e3);
      }
    }
  }
})();

let model: cocoSsd.ObjectDetection | null = null;

import type { YoloResponse, DetectionResult } from '../types';

const classNames = ['No Endodontic Treatment', 'Incomplete Endodontic Treatment', 'Complete Endodontic Treatment', 'Total Endodontic Failure'];


export const detectObjects = async (file: File, enableFilter: boolean = false, enableGradCam: boolean = false): Promise<YoloResponse> => {
  try {
    // Convert File to Base64
    const toBase64 = (file: File): Promise<string> => new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = error => reject(error);
    });

    const base64Image = await toBase64(file);

    console.log('Calling server-side YOLO inference...');

    // Configurable endpoint
    const endpoint = 'http://localhost:5000/detect';

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: base64Image,
        enable_filter: enableFilter,
        enable_gradcam: enableGradCam
      }),
    });

    if (response.ok) {
      const result = await response.json();
      console.log('Server detections:', result.detections);

      return {
        detections: result.detections || [],
        image_size: result.image_size,
        heatmap: result.heatmap
      };
    } else {
      const errorText = await response.text();
      console.error(`Server inference failed (${response.status}):`, errorText);
      throw new Error(`Server error: ${response.statusText}`);
    }
  } catch (error) {
    console.error('Server inference error:', error);
    // Fallback or rethrow
    throw error;
  }
};

