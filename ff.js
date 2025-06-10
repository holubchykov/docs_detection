// import * as ort from "onnxruntime-web";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import { renderBoxes } from "./renderBox";
import labels from "./labels.json";

let session = null;

/**
 * @param {string} url
 * @returns {Promise<void>}
 */
export async function initModel(url) {
  try {
    session = await ort.InferenceSession.create(url);
  } catch (error) {
    console.error("Failed to initialize model:", error);
    throw error;
  }
}

/**
 * Preprocess an image or video source for model input
 * @param {HTMLImageElement|HTMLVideoElement} source - The source element
 * @returns {Array} - [tensorInput, xRatio, yRatio]
 */
function preprocess(source) {
  let xRatio, yRatio;
  const input = tf.tidy(() => {
    const img = tf.browser.fromPixels(source);
    const [h, w] = img.shape;
    const maxSize = Math.max(w, h);

    const padded = tf.pad(img, [
      [0, maxSize - h],
      [0, maxSize - w],
      [0, 0],
    ]);
    xRatio = maxSize / 320;
    yRatio = maxSize / 320;

    const resized = tf.image.resizeBilinear(padded, [320, 320]);
    const normalized = tf.div(resized, 255.0);

    return tf.expandDims(normalized, 0);
  });
  return [input, xRatio, yRatio];
}

/**
 * Calculate Intersection over Union (IoU) between two boxes
 * @param {Array<number>} boxA - First box [x1, y1, x2, y2]
 * @param {Array<number>} boxB - Second box [x1, y1, x2, y2]
 * @returns {number} IoU value between 0 and 1
 */
function calculateIoU(boxA, boxB) {
  // Calculate intersection area
  const xA = Math.max(boxA[0], boxB[0]);
  const yA = Math.max(boxA[1], boxB[1]);
  const xB = Math.min(boxA[2], boxB[2]);
  const yB = Math.min(boxA[3], boxB[3]);
  
  // Return 0 if there is no intersection
  if (xB < xA || yB < yA) return 0;
  
  const intersectionArea = (xB - xA) * (yB - yA);
  
  // Calculate union area
  const boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
  const boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
  
  const unionArea = boxAArea + boxBArea - intersectionArea;
  
  return intersectionArea / unionArea;
}

/**
 * Apply Non-Maximum Suppression to filter overlapping boxes
 * @param {Array<Object>} boxes - Array of box objects with bounding and probability properties
 * @param {number} iouThreshold - IoU threshold for considering boxes as overlapping
 * @param {number} scoreThreshold - Minimum score threshold for boxes
 * @returns {Array<number>} Array of indices of kept boxes
 */
function nonMaxSuppression(boxes, iouThreshold = 0.5, scoreThreshold = 0.3) {
  const filteredBoxes = boxes.filter(box => box.probability >= scoreThreshold);
  
  if (filteredBoxes.length === 0) {
    return [];
  }
  
  // Sort boxes by score in descending order
  const sortedIndices = Array.from({ length: filteredBoxes.length }, (_, i) => i)
    .sort((a, b) => filteredBoxes[b].probability - filteredBoxes[a].probability);
  
  const selectedIndices = [];
  
  while (sortedIndices.length > 0) {
    const currentIdx = sortedIndices.shift();
    selectedIndices.push(currentIdx);
    
    // Compare current box with all remaining boxes
    for (let i = sortedIndices.length - 1; i >= 0; i--) {
      const idx = sortedIndices[i];
      
      const iou = calculateIoU(
        filteredBoxes[currentIdx].bounding,
        filteredBoxes[idx].bounding
      );
      
      // Remove boxes with IoU greater than threshold
      if (iou >= iouThreshold) {
        sortedIndices.splice(i, 1);
      }
    }
  }
  
  return selectedIndices.map(idx => filteredBoxes[idx]);
}

/**
 * @param {HTMLImageElement|HTMLVideoElement} sourceEl - Source element (image or video)
 * @param {HTMLCanvasElement} canvasEl - Canvas element to draw results
 * @param {Function} callback - Callback function with highConfidence boolean parameter
 * @returns {Promise<void>}
 */
export async function detect(sourceEl, canvasEl, callback = () => {}) {
  if (!session) {
    console.error("ONNX session not initialized");
    return;
  }

  try {
    const MODEL_W = 320, MODEL_H = 320;
    const [input, xRatio, yRatio] = preprocess(sourceEl);

    const tmp = document.createElement("canvas");
    tmp.width = MODEL_W; tmp.height = MODEL_H;
    const tmpCtx = tmp.getContext("2d");
    tmpCtx.drawImage(sourceEl, 0, 0, MODEL_W, MODEL_H);
    const { data: imgData } = tmpCtx.getImageData(0, 0, MODEL_W, MODEL_H);

    const tensorData = new Float16Array(1 * 3 * MODEL_H * MODEL_W);
    for (let y = 0; y < MODEL_H; ++y) {
      for (let x = 0; x < MODEL_W; ++x) {
        const i = (y * MODEL_W + x) * 4;
        const idx = y * MODEL_W + x;
        tensorData[idx] = imgData[i] / 255;                   // R
        tensorData[MODEL_H * MODEL_W + idx] = imgData[i+1] / 255; // G
        tensorData[2 * MODEL_H * MODEL_W + idx] = imgData[i+2] / 255; // B
      }
    }
    const inputTensor = new ort.Tensor("float16", tensorData, [1, 3, MODEL_H, MODEL_W]);

    const feeds = { [session.inputNames[0]]: inputTensor };
    const results = await session.run(feeds);
    const output = results[session.outputNames[0]];
    
    const numBoxes = output.dims[1];     // detections
    const numClass = output.dims[2] - 4;  // classes per box

    const detectionBoxes = [];
    
    for (let idx = 0; idx < output.dims[1]; idx++) {
      const data = output.data.slice(idx * output.dims[2], (idx + 1) * output.dims[2]); // get rows
      let box = data.slice(0, 4);
      const scores = data.slice(4, 4 + numClass);
      const score = Math.max(...scores);
      const label = scores.indexOf(score);
      const [x1, y1, x2, y2] = box;

      detectionBoxes.push({
        label: labels[label],
        probability: score,
        bounding: [x1, y1, x2, y2],
      });
    }
    
    if (detectionBoxes.length === 0) {
      callback(false);
      return;
    }
    
    const nmsThreshold = 0.5; // IoU threshold for NMS
    const scoreThreshold = 0.8; // Minimum score to consider
    const filteredBoxes = nonMaxSuppression(detectionBoxes, nmsThreshold, scoreThreshold);
    
    if (filteredBoxes.length === 0) {
      callback(false);
      return;
    }
    
    const origW = sourceEl instanceof HTMLVideoElement
      ? sourceEl.videoWidth
      : sourceEl.naturalWidth;
    const origH = sourceEl instanceof HTMLVideoElement
      ? sourceEl.videoHeight
      : sourceEl.naturalHeight;
    const xScale = origW / MODEL_W;
    const yScale = origH / MODEL_H;

    const finalBoxes = [];
    const finalScores = [];
    const finalClasses = [];
    
    filteredBoxes.forEach(box => {
      const [x1, y1, x2, y2] = box.bounding;
      
      finalBoxes.push([
        Math.max(0, Math.round(x1 * xScale)),
        Math.max(0, Math.round(y1 * yScale)),
        Math.min(origW, Math.round(x2 * xScale)),
        Math.min(origH, Math.round(y2 * yScale)),
      ]);
      
      finalScores.push(box.probability);
      finalClasses.push(box.label);
    });

    const ctx = canvasEl.getContext("2d");
    
    // Set canvas dimensions to match the source element
    canvasEl.width = origW;
    canvasEl.height = origH;
    
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    // Find high confidence detection
    const idxHigh = finalScores.findIndex(s => s >= 0.95);
    if (idxHigh >= 0) {
      const [x1, y1, x2, y2] = finalBoxes[idxHigh];
      const boxW = x2 - x1, boxH = y2 - y1;
      const areaRatio = (boxW * boxH) / (origW * origH);

      if (areaRatio >= 0.5) {
        const scaleFactor = 1.2;
        const newWidth = boxW * scaleFactor;
        const newHeight = boxH * scaleFactor;

        canvasEl.width = newWidth;
        canvasEl.height = newHeight;
        canvasEl.style.width = `${newWidth}px`;
        canvasEl.style.height = "auto";

        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(sourceEl, x1, y1, boxW, boxH, 0, 0, newWidth, newHeight);
        
        ctx.fillStyle = "red";
        ctx.font = "20px Arial";
        ctx.fillText(`Doc: ${(areaRatio*100).toFixed(1)}%`, 10, 25);
        
        if (sourceEl.srcObject) {
          sourceEl.srcObject.getTracks().forEach(t => t.stop());
        }
        callback(true);
        return;
      }
    }

    renderBoxes(canvasEl, finalBoxes, finalScores, finalClasses, [xScale, yScale]);
    
    if (idxHigh >= 0) {
      const [x1, y1, x2, y2] = finalBoxes[idxHigh];
      const areaRatio = ((x2 - x1) * (y2 - y1)) / (origW * origH);
      ctx.fillStyle = "red";
      ctx.font = "20px Arial";
      ctx.fillText(`Doc: ${(areaRatio*100).toFixed(1)}%`, 10, 25);
    }
    callback(false);
  } catch (error) {
    console.error("Error during detection:", error);
    callback(false);
  }
}

/**
 * @param {HTMLVideoElement} vidSource - Video source element 
 * @param {HTMLCanvasElement} canvasEl - Canvas element to draw results
 */
export const detectVideo = (vidSource, canvasEl) => {
  let frameCount = 0;
  let animationId;
  
  const loop = async () => {
    frameCount++;
    if (frameCount % 10 === 0) {
      await detect(vidSource, canvasEl, highConfidence => {
        if (!highConfidence) {
          animationId = requestAnimationFrame(loop);
        }
      });
    } else {
      animationId = requestAnimationFrame(loop);
    }
  };
  
  // Start the detection loop
  loop();
  
  // Return function to stop detection if needed
  return () => {
    if (animationId) {
      cancelAnimationFrame(animationId);
    }
  };
};
