// data_upload.js

$(document).ready(function() {

  // When a file is selected
  $('#data-file').on('change', function() {
    const file = this.files[0];
    if (!file) return;

    $(this).prop('disabled', true);

    const formData = new FormData();
    formData.append('data_file', file);

    $.ajax({
      url: '/upload',
      type: 'POST',
      data: formData,
      processData: false,
      contentType: false,
      success: function(response) {
        $('#clear-data').prop('disabled', false);
        $('.vstep').removeClass('hidden');
      },
      error: function(err) {
        console.error(err);
        alert('Upload failed');
        $('#data-file').prop('disabled', false);
      }
    });
  });

  // Clear data
  $('#clear-data').on('click', function() {
    $.ajax({
      url: '/clear_data',
      type: 'POST',
      success: function() {
        $('#data-file').prop('disabled', false).val('');
        $('#clear-data').prop('disabled', true);
        $('.vstep').addClass('hidden');
      }
    });
  });

});

