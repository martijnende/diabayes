$(document).ready(function() {

  // Update request
  $('#vstep-container').on('click', 'button.action-btn', function() {
    const action = $(this).data('action');
    const form = $(this).closest('form');
    const formData = form.serialize() + '&action=' + action;

    $.ajax({
      url: '/update-step',
      type: 'POST',
      data: formData,
      success: function(response) {
        console.log("Update success");
        if (response.html) {
          $("div#vstep-container").html(response.html);
        }
      },
      error: function(err) {
        console.error(err);
      }
    });
  });

  // Insert request
  $(document).on('click', 'button#add-vstep', function() {
    $.ajax({
      url: '/update-step',
      type: 'POST',
      data: 'action=add',
      success: function(response) {
        if (response.html) {
          $("div#vstep-container").html(response.html);
        }
      },
      error: function(err) {
        console.error(err);
      }
    });
  });
});

