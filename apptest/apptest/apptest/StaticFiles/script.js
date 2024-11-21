// Ensure functions are in the global scope
window.loadMetrics = loadMetrics;
window.loadNextPrice = loadNextPrice;

// Function to load metrics from the API
async function loadMetrics() {
    console.log("loadMetrics called");
    try {
        const response = await fetch('/metrics');
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        const data = await response.json();
        document.getElementById('mse').innerText = `MSE: ${data.mse}`;
        document.getElementById('r2').innerText = `R²: ${data.r2}`;
    } catch (error) {
        console.error('Error fetching metrics: ', error.message);
    }
}

// Function to load the next predicted price and display it
async function loadNextPrice() {
    try {
        const response = await fetch('/next-price');
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        const data = await response.json();
        document.getElementById('next-price-value').innerText =
            `Next Price: ${data.nextPrice.toFixed(2)} on ${data.nextDate}`;
    } catch (error) {
        alert('Error fetching next price: ' + error.message);
    }
}


// Function to upload a CSV file to the server
async function uploadCSV() {
    const fileInput = document.getElementById('file-input');
    const statusElement = document.getElementById('upload-status');

    // Clear any previous status messages
    statusElement.innerText = '';

    // Check if a file is selected
    if (fileInput.files.length === 0) {
        statusElement.innerText = 'Please select a file to upload.';
        return;
    }

    // Get the selected file
    const file = fileInput.files[0];

    // Prepare the FormData object
    const formData = new FormData();
    formData.append('file', file);

    try {
        // Send the file to the server
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        // Check the server response
        if (response.ok) {
            const result = await response.text();
            statusElement.innerText = `Upload successful: ${result}`;
        } else {
            const error = await response.text();
            statusElement.innerText = `Upload failed: ${error}`;
        }
    } catch (error) {
        statusElement.innerText = `Error uploading file: ${error.message}`;
    }
}



// Function to load predictions from the API and display them in a table
async function loadPredictions() {
    try {
        const response = await fetch('/predictions');

        // Check if the response is OK
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        const table = document.getElementById('predictions-table');
        table.innerHTML = ''; // Clear previous rows

        // Populate the table with prediction data
        data.forEach(pred => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${pred.actual.toFixed(2)}</td>
                <td>${pred.predicted.toFixed(2)}</td>`;
            table.appendChild(row);
        });
    } catch (error) {
        alert('Error fetching predictions: ' + error.message);
    }
}


