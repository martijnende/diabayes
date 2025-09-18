document.addEventListener("DOMContentLoaded", function() {
  let loadingOverlay = document.getElementById("loading-overlay");
  let mainContent = document.getElementById("main-content");

  // Hide loading overlay and fade in main content
  setTimeout(() => {
    loadingOverlay.style.opacity = "0";
    loadingOverlay.style.transition = "opacity 0.5s ease-out";

    setTimeout(() => {
      loadingOverlay.style.display = "none";
      mainContent.style.display = "block";
      mainContent.style.opacity = "0";
      mainContent.style.transition = "opacity 0.5s ease-in";
      resizePlotlyGraphs();
      setTimeout(() => {
        mainContent.style.opacity = "1";
        mainContent.classList.remove("invisible");
      }, 1);
    }, 1);
  }, 1); // Small delay for smoother effect
});
