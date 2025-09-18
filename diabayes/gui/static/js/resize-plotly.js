function resizePlotlyGraphs() {
  if (Plotly) {
    document.querySelectorAll('.js-plotly-plot').forEach(plot => {
      Plotly.relayout(plot, { autosize: true });
    });
  }
}

// Call resizePlotlyGraphs() on window resize
window.addEventListener("resize", function() {
  setTimeout(resizePlotlyGraphs, 200); // Small delay to avoid excessive calls
});

