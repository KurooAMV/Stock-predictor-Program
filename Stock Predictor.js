document.getElementById("stockPredictionForm").addEventListener("submit", function (e) {
    e.preventDefault();
    
    const lookBack = document.getElementById("lookBack").value;
    
    // Make an API request to your backend with the selected look-back period
    fetch(`/predict?lookBack=${lookBack}`, {
        method: "GET",
        headers: {
            "Content-Type": "application/json",
        },
    })
    .then(response => response.json())
    .then(data => {
        // Display the prediction result
        document.getElementById("predictionResult").innerHTML = `
            <h2>Prediction:</h2>
            <p>Next day's stock price: $${data.prediction.toFixed(2)}</p>
        `;
    })
    .catch(error => {
        console.error("Error:", error);
    });
});
