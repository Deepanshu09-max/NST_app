document.addEventListener('DOMContentLoaded', () => {
    console.log('script.js loaded and DOM ready');
    const form = document.getElementById('nst-form');
    const resultSection = document.getElementById('result-section');
    const stylizedImg = document.getElementById('stylized-image');
    const downloadLink = document.getElementById('download-link');
    const feedbackForm = document.getElementById('feedback-form');
    const contentImageInput = document.getElementById('content-image');
    const contentPreview = document.getElementById('content-preview');

    let currentModel = '';
    let currentContentFilename = '';
    let currentOutputFilename = '';

    contentImageInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                contentPreview.src = e.target.result;
                contentPreview.style.display = 'block';
            }
            reader.readAsDataURL(file);
        } else {
            contentPreview.src = '#';
            contentPreview.style.display = 'none';
        }
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        resultSection.style.display = 'none';
        stylizedImg.src = '';
        downloadLink.href = '#';
        downloadLink.removeAttribute('download');
        currentModel = '';
        currentContentFilename = '';
        currentOutputFilename = '';

        const model = document.getElementById('model').value;
        const contentImage = contentImageInput.files[0];

        if (!contentImage) {
            alert('Please select a content image.');
            return;
        }

        const formData = new FormData();
        formData.append('model', model);
        formData.append('content_image', contentImage);

        try {
            const resp = await fetch('/stylize', { // Ensure this matches the backend route
                method: 'POST',
                body: formData
            });

            if (resp.ok) {
                const resultData = await resp.json();
                const base64Image = resultData.stylized_image_base64;
                const outputFilename = resultData.output_filename;

                if (!base64Image) {
                    throw new Error("Received response, but 'stylized_image_base64' field was missing.");
                }

                const dataUri = `data:image/png;base64,${base64Image}`;
                stylizedImg.src = dataUri;
                downloadLink.href = dataUri;

                currentModel = model;
                currentContentFilename = contentImage.name;
                currentOutputFilename = outputFilename;
                downloadLink.download = currentOutputFilename;

                resultSection.style.display = 'block';
            } else {
                const errorData = await resp.json().catch(() => ({ detail: 'Failed to parse error response.' }));
                console.error('Stylization failed:', errorData);
                alert(`Stylization failed: ${errorData.detail || errorData.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Network or fetch error:', error);
            alert(`Error during stylization request: ${error.message}`);
        }
    });

    feedbackForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const feedbackValue = document.getElementById('feedback').value;

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
            const feedbackResp = await fetch('/feedback', { // Ensure this matches the backend route
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(feedbackData)
            });

            if (feedbackResp.ok) {
                alert('Feedback submitted!');
            } else {
                const errorData = await feedbackResp.json().catch(() => ({ detail: 'Failed to parse error response.' }));
                console.error('Feedback submission failed:', errorData);
                alert(`Failed to submit feedback: ${errorData.detail || errorData.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Network or fetch error during feedback:', error);
            alert(`Error submitting feedback: ${error.message}`);
        }
    });
});
