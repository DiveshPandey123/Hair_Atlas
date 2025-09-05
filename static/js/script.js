
// Global variables for prediction functionality (used on index.html)
let currentFaceShape = null;
let lastSuggestions = [];
let stream = null;
let capturedPhoto = null;

// Flag to track if the imageInput event listener has been added
let isImageInputListenerAdded = false;

// Global dropdown toggle functionality (works on all pages)
document.addEventListener('DOMContentLoaded', function() {
    // Dropdown menu functionality
    const dropdowns = document.querySelectorAll('.dropdown');
    dropdowns.forEach(dropdown => {
        const trigger = dropdown.querySelector('a');
        const dropdownContent = dropdown.querySelector('.dropdown-content');

        if (trigger && dropdownContent) {
            // Toggle dropdown on click
            trigger.addEventListener('click', function(event) {
                event.preventDefault();
                event.stopPropagation();
                // Close all other dropdowns
                document.querySelectorAll('.dropdown-content').forEach(content => {
                    if (content !== dropdownContent) {
                        content.classList.remove('show');
                    }
                });
                // Toggle the current dropdown
                dropdownContent.classList.toggle('show');
            });

            // Close dropdown when clicking outside
            document.addEventListener('click', function(e) {
                if (!dropdown.contains(e.target)) {
                    dropdownContent.classList.remove('show');
                }
            });
        }
    });

    // Theme toggle functionality
    const themeToggle = document.querySelector('.theme-toggle i');
    if (themeToggle) {
        if (localStorage.getItem('dark-theme') === 'enabled') {
            document.body.classList.add('dark-theme');
            themeToggle.classList.replace('fa-sun', 'fa-moon');
        }

        themeToggle.addEventListener('click', function() {
            document.body.classList.toggle('dark-theme');
            if (document.body.classList.contains('dark-theme')) {
                themeToggle.classList.replace('fa-sun', 'fa-moon');
                localStorage.setItem('dark-theme', 'enabled');
            } else {
                themeToggle.classList.replace('fa-moon', 'fa-sun');
                localStorage.setItem('dark-theme', 'disabled');
            }
        });
    }
});

// Page-specific functionality for index.html
if (document.getElementById('getStartedBtn')) {
    // "Get Started" button functionality (only on index.html)
    const getStartedBtn = document.getElementById('getStartedBtn');
    const mainContent = document.querySelector('.main-content');

    if (getStartedBtn && mainContent) {
        getStartedBtn.addEventListener('click', function() {
            mainContent.classList.remove('hidden');
            getStartedBtn.parentElement.style.display = 'none'; // Hide the hero section
        });
    }

    // Prediction form functionality (only on index.html)
    function previewImage(event) {
        const imagePreview = document.getElementById('imagePreview');
        // Clear the preview to avoid duplicates
        imagePreview.innerHTML = '';
        const file = event.target.files[0];
        if (file) {
            capturedPhoto = file;
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.classList.add('preview-img');
                imagePreview.appendChild(img);
            };
            reader.readAsDataURL(file);
        }
    }

    async function startCamera() {
        const video = document.getElementById('video');
        const cameraSection = document.getElementById('cameraSection');
        const imagePreview = document.getElementById('imagePreview');
        const imageInput = document.getElementById('imageInput');

        imageInput.value = '';
        imagePreview.innerHTML = '';
        capturedPhoto = null;

        cameraSection.classList.remove('hidden');

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user' }
            });
            video.srcObject = stream;
        } catch (error) {
            console.error("Error accessing camera:", error);
            alert("Could not access the camera. Please ensure you have a camera and have granted permission.");
            cameraSection.classList.add('hidden');
        }
    }

    function capturePhoto() {
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const imagePreview = document.getElementById('imagePreview');
        const cameraSection = document.getElementById('cameraSection');

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob((blob) => {
            capturedPhoto = blob;

            const img = document.createElement('img');
            img.src = URL.createObjectURL(blob);
            img.classList.add('preview-img');
            imagePreview.innerHTML = '';
            imagePreview.appendChild(img);

            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            cameraSection.classList.add('hidden');
        }, 'image/jpeg');
    }

    function stopCamera() {
        const cameraSection = document.getElementById('cameraSection');
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        cameraSection.classList.add('hidden');
    }

    document.getElementById('predictButton').addEventListener('click', async (event) => {
        event.preventDefault();
        const fileInput = document.getElementById('imageInput');
        const manualFaceShape = document.getElementById('manualFaceShape').value;
        const resultText = document.getElementById('resultText');
        const suggestionsElement = document.getElementById('suggestions');
        const loadingSpinner = document.getElementById('loadingSpinner');

        loadingSpinner.classList.remove('hidden');
        suggestionsElement.innerHTML = '';
        resultText.innerHTML = '';

        if (manualFaceShape) {
            try {
                const response = await fetch('/predict_manual', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ face_shape: manualFaceShape })
                });
                const data = await response.json();
                loadingSpinner.classList.add('hidden');
                if (data.error) {
                    resultText.innerHTML = `Error: ${data.error}`;
                    return;
                }
                currentFaceShape = data.face_shape;
                lastSuggestions = data.hairstyle_suggestions || [];
                console.log("Manual Prediction Response:", data);
                resultText.innerHTML = `Selected Face Shape: ${currentFaceShape}`;
                displaySuggestions(lastSuggestions);
            } catch (error) {
                loadingSpinner.classList.add('hidden');
                resultText.innerHTML = `Error: ${error.message}`;
                console.error("Manual Prediction Error:", error);
            }
        } else {
            if (!capturedPhoto && !fileInput.files[0]) {
                loadingSpinner.classList.add('hidden');
                resultText.innerHTML = "Error: Please upload an image or capture a photo.";
                return;
            }

            const formData = new FormData();
            if (capturedPhoto) {
                formData.append('file', capturedPhoto, 'captured_photo.jpg');
            } else {
                formData.append('file', fileInput.files[0]);
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                loadingSpinner.classList.add('hidden');
                if (data.error) {
                    resultText.innerHTML = `Error: ${data.error}`;
                    return;
                }
                currentFaceShape = data.face_shape;
                lastSuggestions = data.hairstyle_suggestions || [];
                console.log("Prediction Response:", data);
                resultText.innerHTML = `Predicted Face Shape: ${currentFaceShape}`;
                displaySuggestions(lastSuggestions);
            } catch (error) {
                loadingSpinner.classList.add('hidden');
                resultText.innerHTML = `Error: ${error.message}`;
                console.error("Prediction Error:", error);
            }
        }
    });

    document.getElementById('applyFilterButton').addEventListener('click', (event) => {
        event.preventDefault();
        const lengthFilter = document.getElementById('hairLengthFilter').value;
        const typeFilter = document.getElementById('hairTypeFilter').value;
        const occasionFilter = document.getElementById('occasionFilter').value;

        if (!currentFaceShape) {
            document.getElementById('suggestions').innerHTML = "Please predict or select a face shape first.";
            return;
        }

        let filteredSuggestions = [...lastSuggestions].filter(suggestion => {
            let matches = true;
            if (lengthFilter && !suggestion.name.toLowerCase().includes(lengthFilter.toLowerCase())) {
                matches = false;
            }
            if (typeFilter && !suggestion.name.toLowerCase().includes(typeFilter.toLowerCase())) {
                matches = false;
            }
            return matches;
        });

        if (occasionFilter) {
            filteredSuggestions = filteredSuggestions.map(s => ({
                name: `${s.name} for ${occasionFilter}`,
                image: s.image
            }));
        }

        if (filteredSuggestions.length === 0) {
            filteredSuggestions = [{ name: "No suggestions match your filters", image: "" }];
        }

        displaySuggestions(filteredSuggestions);
    });

    function displaySuggestions(suggestions) {
        const suggestionsDiv = document.getElementById('suggestions');
        if (!suggestions || suggestions.length === 0) {
            suggestionsDiv.innerHTML = "<p>No suggestions available.</p>";
            return;
        }
        suggestionsDiv.innerHTML = suggestions.map(s => {
            console.log("Rendering suggestion:", s);
            const name = s && s.name ? s.name : "Unnamed Hairstyle";
            const image = s && s.image ? s.image : "";
            return `
                <div class="suggestion-item">
                    ${image ? `<img src="${image}" alt="${name}" class="hairstyle-image" onerror="this.src='https://via.placeholder.com/150'; this.alt='Image not found'">` : ''}
                    <p>${name}</p>
                    ${image ? `<form method="post" action="/favorites">
                        <input type="hidden" name="hairstyle" value="${name}">
                        <button type="submit">Add to Favorites</button>
                    </form>` : ''}
                </div>
            `;
        }).join('');
    }

    // Attach event listeners for index.html elements
    const imageInput = document.getElementById('imageInput');
    if (imageInput && !isImageInputListenerAdded) {
        imageInput.addEventListener('change', previewImage);
        isImageInputListenerAdded = true; // Set the flag to true after adding the listener
    }

    const startCameraButton = document.getElementById('startCameraButton');
    if (startCameraButton) {
        startCameraButton.addEventListener('click', startCamera);
    }

    const captureButton = document.getElementById('captureButton');
    if (captureButton) {
        captureButton.addEventListener('click', capturePhoto);
    }

    const stopCameraButton = document.getElementById('stopCameraButton');
    if (stopCameraButton) {
        stopCameraButton.addEventListener('click', stopCamera);
    }
}

// document.addEventListener('DOMContentLoaded', () => {
//     // Only run on hairstyles.html
//     if (document.getElementById('femaleGallery') && document.getElementById('maleGallery')) {
//         const PEXELS_API_KEY = 'cnMr1fm7oGDTKWJmZg9j2zBttAD63OnNANTMFJorXgtrxkPaEoMp5sqz';  // Replace this with your actual Pexels key

//         async function fetchHairstyles(query, count = 15) {
//             const response = await fetch(`https://api.pexels.com/v1/search?query=${query}&per_page=${count}`, {
//                 headers: {
//                     Authorization: PEXELS_API_KEY
//                 }
//             });
//             const data = await response.json();
//             return data.photos || [];
//         }

//         function displayHairstyles(photos, containerId) {
//             const container = document.getElementById(containerId);
//             container.innerHTML = ''; // clear existing items
//             photos.forEach(photo => {
//                 const div = document.createElement('div');
//                 div.classList.add('gallery-item');
//                 div.innerHTML = `
//                     <img src="${photo.src.medium}" alt="${photo.photographer}" />
//                     <p>${photo.alt || 'Hairstyle'}</p>
//                 `;
//                 container.appendChild(div);
//             });
//         }

//         async function loadHairstyles() {
//             try {
//                 const femalePhotos = await fetchHairstyles('female hairstyle');
//                 const malePhotos = await fetchHairstyles('male hairstyle');

//                 displayHairstyles(femalePhotos, 'femaleGallery');
//                 displayHairstyles(malePhotos, 'maleGallery');
//             } catch (err) {
//                 console.error('Failed to load hairstyles:', err);
//             }
//         }

//         loadHairstyles();
//     }
// });
