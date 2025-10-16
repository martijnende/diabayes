$(document).ready(function() {

  var socket = io("/logs");

  function renderLogEntry(entry) {
    return `<div class="log-entry">[${entry.timestamp}] ${entry.level}: ${entry.msg}</div>`;
  }

  function loadLogs() {
    $.getJSON("/logs/all", function(data) {
        const container = $("#log-container");
        container.empty();
        data.forEach(entry => {
          container.append(renderLogEntry(entry));
      });
    });
  }

  // Request logs when the page loads
  loadLogs();

  socket.on("connect", () => {
      console.log("Connected to /logs namespace");
  });

  // Update logs when a new log entry is received
  socket.on("new_log", function(entry) {
    console.log("Entry received");
    $("#log-container").prepend(renderLogEntry(entry));
  });

});
