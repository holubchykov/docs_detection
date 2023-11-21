import logo from './logo.svg';
import './App.css';
import * as tf from '@tensorflow/tfjs';
import React, {useEffect, useRef, useState} from 'react';

tf.setBackend('webgl');
const classes_dir = {
    1: {
        name: 'id/pass',
        id: 1
    }
};

function App() {
    const [model, setModel] = useState();
    const [text, setText] = useState('');
    const [text1, setText1] = useState('');
    const canvasRef = useRef();
    const videoRef = useRef();
    let img;

    //useEffect for loading the model and warming it up...
    useEffect(() => {
        tf.loadGraphModel("https://cdn.jsdelivr.net/gh/nirchetrit/IdPassportDetection@latest/src/model/model.json").then(model => {
            setModel(model);
            console.log('loaded the model');
            console.log('warming up..');
            model.executeAsync(tf.zeros([1, 320, 320, 3]).asType('int32')).then(() => {
                console.log('finish warming up');
                setText(tf.getBackend());
                setText1('you can detect now');
            });
        });
    }, []);


    useEffect(() => {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({
                audio: false,
                video: {
                    facingMode: {
                        exact: 'environment'
                    }
                }
            }).then(stream => {
                window.stream = stream;
                videoRef.current.srcObject = stream;
                if (model && videoRef.current) {
                    console.log('starting the detection');

                }
            });
        }
    }, []);
    

    const getDetectedObjFromPredictions = (threshold, scores, boxes, classes, classesDir) => {
        const detectedObj = [];
        const video_frame = document.getElementById('frame');
        if (scores[0] > threshold) {
            const bbox = [];
            const minY = boxes[0] * video_frame.offsetHeight;
            const minX = boxes[1] * video_frame.offsetWidth;
            const maxY = boxes[2] * video_frame.offsetHeight;
            const maxX = boxes[3] * video_frame.offsetWidth;
            bbox[0] = minX;
            bbox[1] = minY;
            bbox[2] = maxX - minX;
            bbox[3] = maxY - minY;
            detectedObj.push({
                class: classes_dir[classes[0]].name,
                score: scores[0].toFixed(2),
                bbox: bbox
            });
        }
        return detectedObj;
    };
    // function calculateBlur(image) {
    //     // Create a temporary canvas
        
    //     let tempCanvas = createCanvas(image.width, image.height);
    //     let tempContext = tempCanvas.drawingContext;
        
    //     // Draw the image on the temporary canvas
    //     tempContext.drawImage(image, 0, 0, image.width, image.height);
        
    //     // Get the image data
    //     let imageData = tempContext.getImageData(0, 0, image.width, image.height).data;
        
    //     let sum = 0;
      
    //     // Iterate through pixel values
    //     for (let i = 0; i < imageData.length; i += 4) {
    //       // Convert RGB to grayscale
    //       let gray = (imageData[i] + imageData[i + 1] + imageData[i + 2]) / 3;
          
    //       // Accumulate the squared difference between adjacent pixels
    //       sum += Math.pow(gray - imageData[i], 2);
    //     }
      
    //     // Calculate the average squared difference
    //     let blurValue = sum / (image.width * image.height);
      
    //     return blurValue;
    //   }
    const renderPredictions = (predictions, show) => {
        const ctx = canvasRef.current.getContext("2d");
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        const video_frame = document.getElementById('frame');

        // Font options.
        const font = "16px sans-serif";
        ctx.font = font;
        ctx.textBaseline = "top";
        if (show) {
            ctx.fillStyle = "#FF0000"; // Red color for the message
            ctx.font = "20px sans-serif";
            ctx.fillText("Point your camera closer", 10, 10); // Adjust the position as needed
        }

        const boxes = predictions[1].dataSync();
        const classes = predictions[2].dataSync();
        const scores = predictions[4].dataSync();


        const detectedObj = getDetectedObjFromPredictions(0.98, scores, boxes, classes, classes_dir);
        detectedObj.forEach(item => {
            const x = item['bbox'][0];
            const y = item['bbox'][1];
            const width = item['bbox'][2];
            const height = item['bbox'][3];

            // Draw the bounding box.
            ctx.strokeStyle = "#00FFFF";
            ctx.lineWidth = 4;
            ctx.strokeRect(x, y, width, height);
            const croppedCanvas = document.createElement('canvas');
            const croppedCtx = croppedCanvas.getContext('2d');
            croppedCanvas.width = width;
            croppedCanvas.height = height;
            croppedCtx.drawImage(video_frame, x, y, width, height, 0, 0, width, height);
            let imageData = croppedCtx.getImageData(x, y, width, height).data;
            var totalLightness = 0;
            var totalSharpness = 0;
            for (var i = 0; i < imageData.length; i += 4) {
                // Extract RGB values
                var r = imageData[i];
                var g = imageData[i + 1];
                var b = imageData[i + 2];

                // Convert RGB to lightness (assuming linear RGB)
                var lightness = (0.299 * r + 0.587 * g + 0.114 * b);

                // Accumulate lightness values
                totalLightness += lightness;
            }
            for (var i = 0; i < imageData.length; i += 4) {
                // Calculate the sharpness using the Sobel operator
                var gx = imageData[i] * 0.3 + imageData[i + 1] * 0.59 + imageData[i + 2] * 0.11;
                totalSharpness += gx * gx;
              }
            var averageSharpness = totalSharpness / (width * height) / 100;
            var averageLightness = totalLightness / (imageData.length / 4);



            // let sum = 0;
          
            // // Iterate through pixel values
            // for (let i = 0; i < imageData.length; i += 4) {
            //   // Convert RGB to grayscale
            //   let gray = (imageData[i] + imageData[i + 1] + imageData[i + 2]) / 3;
              
            //   // Accumulate the squared difference between adjacent pixels
            //   sum += Math.pow(gray - imageData[i], 2);
            // }
          
            // // Calculate the average squared difference
            // let blurValue = sum / (width * height);
            console.log('Blurriness:', averageSharpness);
            console.log('Lightness:', averageLightness);


            // Draw the label background.
            ctx.fillStyle = "#00FFFF";
            const textWidth = ctx.measureText(item["label"] + " " + (100 * item["score"]).toFixed(2) + "%").width;
            const textHeight = parseInt(font, 10); // base 10
            ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
            ctx.fillStyle = "#000000";
            ctx.fillText(item["class"] + " " + (100 * item["score"]) + "% " + 'bluriness: ' +  averageSharpness.toFixed(2) + '; lightness: ' + averageLightness.toFixed(2), x, y);
            if (!show){

            if (averageSharpness>100) {
            if (averageLightness>50){
            // Display the cropped canvas in a separate block under the video stream.
            const croppedCanvasBlock = document.getElementById('croppedCanvasBlock');
            croppedCanvasBlock.innerHTML = '';
            croppedCanvasBlock.appendChild(croppedCanvas);}}}
        });

    };
    const preprocessInput = (input) => {
        const raw = tf.browser.fromPixels(input);
        const expanded = raw.expandDims();
        return expanded;
    };
    const runModel = async (video) => {
        if (!video) {
            return;
        }
        tf.engine().startScope();
        const preprocessedVideo = preprocessInput(video);
        model.executeAsync(preprocessedVideo).then(predictions => {
            const boxes = predictions[1].dataSync();
            const classes = predictions[2].dataSync();
            const scores = predictions[4].dataSync();
            const detectedObj = getDetectedObjFromPredictions(0.98, scores, boxes, classes, classes_dir);
            let show=false;
            if (detectedObj.length > 0) {
                const boundingBox = detectedObj[0].bbox;
                const frameArea = videoRef.current.offsetHeight * videoRef.current.offsetWidth;
                const boundingBoxArea = boundingBox[2] * boundingBox[3];
                const coveragePercentage = (boundingBoxArea / frameArea) * 100;
                if (coveragePercentage < 40) {
                    show=true;
                }
                else {
                    show=false;
                }
            }
            renderPredictions(predictions, show);
        });

        requestAnimationFrame(() => {
            runModel(video);
        });
        tf.engine().endScope();
    };

    return (
        <div className="App">
            <h1>TEST</h1>
            <video
                className="size"
                autoPlay
                playsInline
                muted
                ref={videoRef}
                id="frame"
                style={{
                    position: 'absolute',
                    top: '300px',
                    left: '300px'
                }}

            />
            <canvas
                className="size"
                ref={canvasRef}
                width="800"
                height="800"
                style={{
                    position: 'absolute',
                    top: '300px',
                    left: '300px'
                }}
            />
        <div
            id="croppedCanvasBlock"
            style={{
                position: 'absolute',
                top: '900px',  // Adjust the position as needed
                left: '300px', // Adjust the position as needed
            }}
        />
            <button
                style={{
                    position: 'absolute',
                    top: '100px',
                    left: '100px'
                }}
                onClick={() => {
                    runModel(videoRef.current);
                }}>Click to run
            </button>
        </div>
    );
}

export default App;
