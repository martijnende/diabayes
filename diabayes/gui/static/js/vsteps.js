// vsteps.js

$(document).ready(function() {

  // Update request
  $('button.action-btn').on('click', function() {
    const action = $(this).data('action');
    const form = $(this).closest('form');
    const formData = form.serialize() + '&action=' + action;

    $.ajax({
      url: '/update-step',
      type: 'POST',
      data: formData,
      success: function(response) {
        console.log("Update success");
      },
      error: function(err) {
        console.error(err);
      }
    });

  });

});

