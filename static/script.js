const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const removeImage = document.getElementById('removeImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultsBody = document.getElementById('resultsBody');
const scanLine = document.getElementById('scanLine');

let selectedFile = null;

// Handle click on drop zone
dropZone.addEventListener('click', () => {
    fileInput.click();
});

// Handle file selection
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag and drop handlers
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();

    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadPlaceholder.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        analyzeBtn.disabled = false;
        
        // Hide previous results
        resultsSection.classList.add('hidden');
    };

    reader.readAsDataURL(file);
}

// Remove image handler
removeImage.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
});

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    uploadPlaceholder.classList.remove('hidden');
    previewContainer.classList.add('hidden');
    analyzeBtn.disabled = true;
    resultsSection.classList.add('hidden');
    scanLine.style.display = 'none';
}

// Analyze image handler
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Show loading
    loadingOverlay.classList.remove('hidden');
    analyzeBtn.disabled = true;
    scanLine.style.display = 'block';

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        });

        let data;
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.includes("application/json")) {
            data = await response.json();
        } else {
            const text = await response.text();
            throw new Error(`Server returned non-JSON response: ${text.substring(0, 100)}...`);
        }

        if (!response.ok) {
            throw new Error(data.error || `Server error: ${response.status}`);
        }

        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        alert(`Analysis Error: ${error.message}`);
        scanLine.style.display = 'none';
    } finally {
        loadingOverlay.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
});

function displayResults(data) {
    resultsBody.innerHTML = '';
    
    data.results.forEach((res, index) => {
        const isDefective = res.status.toLowerCase().includes('defective') && !res.status.toLowerCase().includes('non-defective');
        const statusClass = isDefective ? 'defective' : 'normal';
        const textClass = isDefective ? 'text-defective' : 'text-normal';
        const icon = isDefective ? 'alert-triangle' : 'check-circle';
        
        const card = document.createElement('div');
        card.className = `result-item ${statusClass}`;
        card.style.animationDelay = `${index * 0.15}s`;
        
        card.innerHTML = `
            <span class="model-name">${res.model}</span>
            <div class="prediction-label ${textClass}">
                <i data-lucide="${icon}"></i>
                ${res.label}
            </div>
            <div class="conf-row">
                <span>Confidence</span>
                <span>${res.confidence}</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 0%; background: ${isDefective ? 'var(--danger)' : 'var(--success)'}"></div>
            </div>
        `;
        
        resultsBody.appendChild(card);

        // Trigger animation for progress bar
        setTimeout(() => {
            const fill = card.querySelector('.progress-fill');
            fill.style.width = res.confidence_val + '%';
        }, 300 + (index * 150));
    });

    lucide.createIcons();
    resultsSection.classList.remove('hidden');
    
    // Smooth scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function resetApp() {
    resetUpload();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
