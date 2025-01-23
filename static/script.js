function uploadImage() {
    let fileInput = document.getElementById("fileInput");
    if (fileInput.files.length === 0) {
        alert("Please select an image.");
        return;
    }

    let file = fileInput.files[0];
    let formData = new FormData();
    formData.append("file", file);

    let originalImage = document.getElementById("originalImage");
    let processedImage = document.getElementById("processedImage");

    // Show the original image
    let reader = new FileReader();
    reader.onload = function(event) {
        originalImage.src = event.target.result;
    };
    reader.readAsDataURL(file);

    // Send image to backend
    fetch("/upload/", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Error processing image.");
            return;
        }

        // Convert hex data back to an image
        let byteArray = new Uint8Array(data.image_data.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
        let blob = new Blob([byteArray], { type: "image/jpeg" });
        let url = URL.createObjectURL(blob);
        processedImage.src = url;
    })
    .catch(error => {
        console.error("Error:", error);
    });
}
