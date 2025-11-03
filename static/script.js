// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const resultSection = document.getElementById('resultSection');
const loading = document.getElementById('loading');
const resultCard = document.getElementById('resultCard');
const resultIcon = document.getElementById('resultIcon');
const resultClass = document.getElementById('resultClass');
const confidenceScore = document.getElementById('confidenceScore');

// API endpoint
const API_URL = '/api/predict';

// Initialize
uploadBox.addEventListener('click', () => fileInput.click());
uploadBox.addEventListener('dragover', handleDragOver);
uploadBox.addEventListener('dragleave', handleDragLeave);
uploadBox.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
predictBtn.addEventListener('click', handlePredict);
clearBtn.addEventListener('click', handleClear);

function handleDragOver(e) {
    e.preventDefault();
    uploadBox.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }
    
    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('File size should be less than 10MB');
        return;
    }
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        imagePreview.style.display = 'block';
        predictBtn.disabled = false;
        clearBtn.style.display = 'inline-block';
        resultSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

async function handlePredict() {
    const file = fileInput.files[0];
    if (!file) return;
    
    // Show loading
    loading.style.display = 'block';
    resultSection.style.display = 'none';
    predictBtn.disabled = true;
    
    // Create FormData
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Prediction failed');
        }
        
        // Display result
        displayResult(data);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error: ' + error.message);
    } finally {
        loading.style.display = 'none';
        predictBtn.disabled = false;
    }
}

function displayResult(data) {
    const { class: predictedClass, confidence } = data;
    
    // Set icon
    resultIcon.textContent = predictedClass === 'Cat' ? '🐱' : '🐶';
    
    // Set class
    resultClass.textContent = predictedClass;
    
    // Set confidence
    const confidencePercent = (confidence * 100).toFixed(2);
    confidenceScore.textContent = `${confidencePercent}%`;
    
    // Show result
    resultSection.style.display = 'block';
    
    // Change card color based on prediction
    if (predictedClass === 'Cat') {
        resultCard.style.background = 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)';
    } else {
        resultCard.style.background = 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)';
    }
}

function handleClear() {
    fileInput.value = '';
    imagePreview.style.display = 'none';
    resultSection.style.display = 'none';
    predictBtn.disabled = true;
    clearBtn.style.display = 'none';
}

