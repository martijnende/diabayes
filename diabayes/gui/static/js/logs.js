$(document).ready(function() {

  var socket = io("/logs");

  window.renderLogEntry = function(entry) {
    return `<div class="log-entry">[${entry.timestamp}] ${entry.level}: ${entry.msg}</div>`;
  }

  window.loadLogs = function() {
    $.getJSON("/logs/all", function(data) {
        const container = $("#log-container");
        container.empty();
        data.forEach(entry => {
          container.append(renderLogEntry(entry));
      });
    });
  }

  // Request logs when the page loads
  window.loadLogs();

  socket.on("connect", () => {
      console.log("Connected to /logs namespace");
  });

  // Update logs when a new log entry is received
  socket.on("new_log", function(entry) {
    console.log("Entry received");
    $("#log-container").prepend(renderLogEntry(entry));
  });

  // Clear logs
  $('#clear-logs').on('click', function() {
    $.ajax({
      url: '/clear_logs',
      type: 'POST',
      success: function() {
        $('#log-container div.log-entry').remove();
        window.loadLogs();
      }
    });
  });


});
