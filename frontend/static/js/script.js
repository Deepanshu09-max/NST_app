document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('nst-form');
    const resultSection = document.getElementById('result-section');
    const stylizedImg = document.getElementById('stylized-image');
    const downloadLink = document.getElementById('download-link');
    const feedbackForm = document.getElementById('feedback-form');
    const contentImageInput = document.getElementById('content-image');
    const contentPreview = document.getElementById('content-preview'); // Get preview element

    // Variables to store context for feedback
    let currentModel = '';
    let currentContentFilename = '';
    let currentOutputFilename = '';

    // --- Add Event Listener for Content Image Preview ---
    contentImageInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                contentPreview.src = e.target.result;
                contentPreview.style.display = 'block'; // Show preview
            }
            reader.readAsDataURL(file);
        } else {
            // No file selected or selection cancelled
            contentPreview.src = '#';
            contentPreview.style.display = 'none'; // Hide preview
        }
    });
    // --- End of Preview Logic ---

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        // Clear previous results and context
        resultSection.style.display = 'none';
        stylizedImg.src = '';
        downloadLink.href = '#';
        downloadLink.removeAttribute('download');
        // Clear preview for next upload (optional, could keep it)
        // contentPreview.src = '#';
        // contentPreview.style.display = 'none';
        currentModel = '';
        currentContentFilename = '';
        currentOutputFilename = '';

        const model = document.getElementById('model').value;
        const contentImage = contentImageInput.files[0]; // Use variable

        if (!contentImage) {
            alert('Please select a content image.');
            return;
        }

        const formData = new FormData();
        formData.append('model', model);
        formData.append('content_image', contentImage);

        // Add a loading indicator (optional)
        // e.g., form.querySelector('button[type="submit"]').textContent = 'Stylizing...';
        // form.querySelector('button[type="submit"]').disabled = true;

        try {
            const resp = await fetch('http://localhost:8000/api/stylize', { // Ensure this port matches your routing service
                method: 'POST',
                body: formData
            });

            if (resp.ok) {
                // Parse the JSON response from the backend
                const resultData = await resp.json();
                const base64Image = resultData.stylized_image_base64;
                const outputFilename = resultData.output_filename; // Get filename from response

                if (!base64Image) {
                    throw new Error("Received response, but 'stylized_image_base64' field was missing.");
                }

                // Construct the data URI
                const dataUri = `data:image/png;base64,${base64Image}`;

                // Set the image source and download link
                stylizedImg.src = dataUri;
                downloadLink.href = dataUri;

                // Store context for feedback using filename from response
                currentModel = model;
                currentContentFilename = contentImage.name;
                currentOutputFilename = outputFilename; // Use filename from backend
                downloadLink.download = currentOutputFilename; // Set download filename

                resultSection.style.display = 'block';
            } else {
                // Handle errors from the backend
                const errorData = await resp.json().catch(() => ({ detail: 'Failed to parse error response.' }));
                console.error('Stylization failed:', errorData);
                alert(`Stylization failed: ${errorData.detail || errorData.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Network or fetch error:', error);
            alert(`Error during stylization request: ${error.message}`);
        } finally {
            // Remove loading indicator (optional)
            // e.g., form.querySelector('button[type="submit"]').textContent = 'Stylize';
            // form.querySelector('button[type="submit"]').disabled = false;
        }
    });

    feedbackForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const feedbackValue = document.getElementById('feedback').value;

        // Ensure we have context from a successful stylization
        if (!currentModel || !currentContentFilename || !currentOutputFilename) {
            alert('Please stylize an image before submitting feedback.');
            return;
        }

        const feedbackData = {
            feedback: feedbackValue,
            model: currentModel,
            content_image_filename: currentContentFilename,
            output_image_filename: currentOutputFilename,
            timestamp: new Date().toISOString()
        };

        try {
            const resp = await fetch('http://localhost:8000/api/feedback', { // Changed port to 8000
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(feedbackData)
            });

            if (resp.ok) {
                alert('Feedback submitted!');
                // Optionally clear feedback form or hide result section again
            } else {
                const errorData = await resp.json().catch(() => ({ detail: 'Failed to parse error response.' }));
                console.error('Feedback submission failed:', errorData);
                alert(`Failed to submit feedback: ${errorData.detail || errorData.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Network or fetch error during feedback:', error);
            alert(`Error submitting feedback: ${error.message}`);
        }
    });
});
