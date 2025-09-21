// static/js/logs.js
$(document).ready(function() {
    const socket = io({ path: "/socket.io" }).connect("/logs");

    socket.on("new_log", function(data) {
        const row = `<div>[${data.timestamp}] ${data.level}: ${data.msg}</div>`;
        $("pre#log-entries").append(row);
    });
});

