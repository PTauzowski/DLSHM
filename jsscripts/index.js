// Create a simple model.

// import * as tf from '@tensorflow/tfjs';

// Get all the gallery cells
//const galleryCells = document.querySelectorAll('.gallery-image');

function allowDrop(ev) {
  ev.preventDefault();
}

function drag(ev) {
  ev.dataTransfer.setData("Text", ev.target.src);
}

let imageDragged;
function drop(ev) {
  var data = ev.dataTransfer.getData("Text");
  ev.target.src = data;
  segmentImageButton.disabled = false;
  damageDetectButton.disabled = false;
  ev.preventDefault();
  
}

const galleryCells = document.querySelectorAll('td[id^="image"]');
console.log('Gallery cels' + galleryCells);

const imageClick = document.getElementById('image1');
const segmentImageButton = document.getElementById("segmentImage");
const damageDetectButton = document.getElementById("damageDetect");


// Loop through each cell and add a click event listener to the image it contains
galleryCells.forEach(function(cell) {
  cell.addEventListener('click', function() {
    // Your response code here
    console.log('You clicked on the image with src: ' + image.src);
  });
});


const colors = [
	[0,  0,  0],
	[255,0,  0],
	[0,  255,0],
	[0,  0,  255],
	[255,255,0],
	[255,0,  255],
	[0,  255,255],
	[128,0,  0],
];

const colorbar = document.getElementById('colorbar');
colorbar.width = 510;
colorbar.height = 50; 
const lctx = colorbar.getContext("2d");

function displayMaskDescription(mask, x, y, width, height) {

  lctx.setTransform(1, 0, 0, 1, 0, 0);
  const borderSize = 1;
  lctx.strokeStyle = "#000000";
  lctx.lineWidth = borderSize;
//  lctx.strokeRect(borderSize / 2, borderSize / 2, colorbar.width - borderSize, colorbar.height - borderSize);

  const maskNames = [
    "Nonbridge",
    "Slab",
    "Beam",
    "Column",
    "Nonstructural components",
    "Rail",
    "Sleeper",
    "Other components",
  ];

  const color = colors[mask];
  const maskName = maskNames[mask];

  lctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
  lctx.fillRect(x, y-height, width, height);

  lctx.font = "14px Arial";
  lctx.fillStyle = "#000000";
  lctx.fillText(maskName, x+1.5*width, y);

}


function drawOnImage(img = null) {
  const canvasElement = document.getElementById("canvas");
  
  const context = canvasElement.getContext("2d");
  
  // if an image is present,
  // the image passed as a parameter is drawn in the canvas
  if (img) {
    const imageWidth = img.width;
    const imageHeight = img.height;
    // rescaling the canvas element
    canvasElement.width = imageWidth;
    canvasElement.height = imageHeight;
    context.drawImage(img, 0, 0, imageWidth, imageHeight);
  }
}

function loadModel(fileName) {
    return tf.loadLayersModel(fileName);
}

let model;
loadModel('models/ICSHM_RGBbig_L5_E100_aug/model.json').then(function(loadModel) {
	model = loadModel;
});

let modelDMG;
loadModel('models/ICSHM_DMGbig_w1-8-5_L4_E50_aug/model.json').then(function(loadModel) {
	modelDMG = loadModel;
});

let modelDMGC;
loadModel('models/ICSHM_DMGCbig_w1-8_L4_E100aug/model.json').then(function(loadModel) {
	modelDMGC = loadModel;
});


const chooseFiles = document.getElementById('chooseFiles');
const inputImage = document.getElementById('image');


chooseFiles.onchange = () => {
    const [file] = chooseFiles.files
    if (file) {
        inputImage.src = URL.createObjectURL(file);
        segmentImageButton.disabled = false;
        damageDetectButton.disabled = false;
    }
};

segmentImageButton.onclick = predict;
damageDetectButton.onclick = damagePredict;

function updateCanvas(canvas, text) {
	// get the canvas context
	var ctx = canvas.getContext("2d");

	// clear the canvas
	ctx.clearRect(0, 0, canvas.width, canvas.height);

	// calculate the x and y position for the center of the canvas
	var xpos = canvas.width / 2;
	var ypos = canvas.height / 2;

	// write the text on the canvas
	ctx.fillText(text, xpos, ypos);

}

 
async function predict() {
   let	o=10;
   let  h=12
    for (let i=0; i<4; i++) {
	    displayMaskDescription(i, 0, o+i*h, 20, 10)
            displayMaskDescription(i+4, 150, o+i*h, 20, 10)
    }


    canvas = document.getElementById('canvas');
    context = canvas.getContext("2d");
    context.mozImageSmoothingEnabled = true;
    context.webkitImageSmoothingEnabled = true;
    context.msImageSmoothingEnabled = true;
    context.imageSmoothingEnabled = true;

    requestAnimationFrame(function() {
				    updateCanvas(canvas, "Prediction in progress...");
			});

    console.log(inputImage);
    const x = tf.browser.fromPixels(inputImage).asType('float32').div(255.0);
    const inputTensor = x.expandDims(0); 
    let prediction = await model.predict(inputTensor);
    const resizedMasks = tf.squeeze(prediction);
    
    context.imageSmoothingEnabled = true;
//    const masks = tf.image.resizeBilinear( resizedMasks, [320, 640] );
    const masks=resizedMasks;	
    rgbMask = tf.zeros([320, 640,3]);
    rgbOnes = tf.ones([320, 640,3]);
    for (let i=0; i<masks.shape[masks.shape.length -1 ]; i++) {
	    const mask=masks.slice([0, 0, i],[320, 640, 1]);
	    const grayscale=mask.squeeze();
	    const [height, width] = grayscale.shape;
	    const color=colors[i % colors.length];
	    const redTensor=grayscale.mul(color[0]/255.0);
   	    const greenTensor=grayscale.mul(color[1]/255.0);
    	    const blueTensor=grayscale.mul(color[2]/255.0);
    	    rgbMask=tf.add(rgbMask,tf.stack([redTensor, greenTensor, blueTensor],2));
	    context.fillStyle='rgba(${color[0]},${color[1]},${color[2]},0.5)';

    }
//    const pixelData=await tf.browser.toPixels(x.mul(255).asType('int32'),canvas);
    const pixelData=await tf.browser.toPixels(rgbMask.div(rgbMask.max()).mul(255).asType('int32'),canvas);
    const imageData = new ImageData(pixelData, 640, 320);
    updateCanvas(canvas, "Prediction ready");
 //   context.globalCompositeOperation ='destination-in';
    context.putImageData(imageData,0,0); 
   

}

async function damagePredict() {
    canvas = document.getElementById('canvas');
    context = canvas.getContext("2d");

    requestAnimationFrame(function() {
				    updateCanvas(canvas, "Prediction in progress...");
			});

    const x = tf.browser.fromPixels(inputImage).asType('float32').div(255.0)
    const inputTensor = x.expandDims(0); 
    let prediction = await modelDMGC.predict(inputTensor);
    const resizedMasks = tf.squeeze(prediction);
    
    context.imageSmoothingEnabled = true;
    const masks = tf.image.resizeBilinear ( resizedMasks, [320,640] );
    rgbMask = tf.zeros([320, 640,3]);
    rgbOnes = tf.ones([320, 640,3]);
    for (let i=0; i<masks.shape[masks.shape.length -1 ]; i++) {
	    const mask=masks.slice([0, 0, i],[320, 640, 1]);
	    const grayscale=mask.squeeze();
	    const [height, width] = grayscale.shape;
	    const color=colors[i % colors.length];
	    const redTensor=grayscale.mul(color[0]/255.0);
   	    const greenTensor=grayscale.mul(color[1]/255.0);
    	    const blueTensor=grayscale.mul(color[2]/255.0);
    	    rgbMask=tf.add(rgbMask,tf.stack([redTensor, greenTensor, blueTensor],2));
	    context.fillStyle='rgba(${color[0]},${color[1]},${color[2]},0.5)';

    }
    const pixelData=await tf.browser.toPixels(rgbMask.div(rgbMask.max()).mul(255).asType('int32'),canvas);
    const imageData = new ImageData(pixelData,640, 320);
    updateCanvas(canvas, "Prediction ready");
 //   context.globalCompositeOperation ='destination-in';
    context.putImageData(imageData,0,0); 

}


