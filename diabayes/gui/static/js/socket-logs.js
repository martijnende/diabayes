var socket = io.connect("http://" + document.domain + ":" + location.port);

// Request logs when the page loads
socket.emit("request_logs");

// Update logs when a new log entry is received
socket.on("new_log", function(log) {
  let logList = document.getElementById("log-list");
  let newLog = document.createElement("li");
  // newLog.textContent = `[${log.timestamp}] ${log.level}: ${log.message}`;
  newLog.innerHTML = log;
  logList.prepend(newLog);
});

// Update full log list if requested
socket.on("update_logs", function(logs) {
  let logList = document.getElementById("log-list");
  logList.innerHTML = "";  // Clear existing logs
  logs.forEach(log => {
    let logItem = document.createElement("li");
    logItem.innerHTML = log;
    // logItem.textContent = `[${log.timestamp}] ${log.level}: ${log.message}`;
    logList.appendChild(logItem);
  });
});
