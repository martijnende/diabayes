$(document).ready(function () {

  function compute_kc($step) {
    // const $step = $(this).closest('.vstep');
    const names = ['a', 'b', 'Dc', 'k'];
    const values = names.map(name => parseFloat($step.find(`input[name="${name}"]`).val()));
    const allValid = values.every(v => !isNaN(v) && v > 0);
    const $result = $step.find('input[name="kkc"]');

    if (allValid) {
      const kkc = values[3] * values[2] / (values[1] - values[0])
      $result.val(kkc.toFixed(2));
    } else {
      $result.val('');
    }
  }

  // Insert request
  $(document).on('click', 'button#add-vstep', function () {
    $.ajax({
      url: '/update-step',
      type: 'POST',
      data: 'action=add',
      success: function (response) {
        if (response.html) {
          $("div#vstep-container").html(response.html);
          $(".vstep").each(function () {
            const $step = $(this);
            compute_kc($step);
          });
          $('[data-bs-toggle="tooltip"]').tooltip();
        }
      },
      error: function (err) {
        console.error(err);
      }
    });
  });

  // Update/invert/delete request
  $('#vstep-container').on('click', 'button.action-btn', function () {
    const action = $(this).data('action');
    const form = $(this).closest('form');
    const formData = form.serialize() + '&action=' + action;
    const icon = $(this).children('i').first();

    icon.addClass('rotate');

    $.ajax({
      url: '/update-step',
      type: 'POST',
      data: formData,
      success: function (response) {
        console.log("Update success");
        if (response.html) {
          $("div#vstep-container").html(response.html);
          $(".vstep").each(function () {
            const $step = $(this);
            compute_kc($step);
          });
          $('[data-bs-toggle="tooltip"]').tooltip();
        }
      },
      error: function (err) {
        console.error(err);
        icon.removeClass('rotate');
      }
    });

  });

  // Switch theta mode
  $(document).on('change', 'input[name="theta_mode"]', function () {
    const $theta0 = $(this).closest('.input-group').find('input#theta0');
    const custom = $(this).val() === 'custom';
    $theta0.prop('disabled', !custom);
  });

  // Calculate k/kc
  $(document).on('input change', '.vstep input[name="a"], .vstep input[name="b"], .vstep input[name="Dc"], .vstep input[name="k"]', function () {
    const $step = $(this).closest('.vstep');
    compute_kc($step);
  });

  // Copy inverted value
  $(document).on('click', 'i.bi-clipboard', function () {
    const $icon = $(this);
    const $field = $icon.closest('.input-group').find('input');
    const value = $field.val();
    navigator.clipboard.writeText(value).then(() => {
      $icon.removeClass('bi-clipboard').addClass('bi-clipboard-check');
      // Revert after 2 sec
      setTimeout(() => {
        $icon.removeClass('bi-clipboard-check').addClass('bi-clipboard');
      }, 2000);
    });
  });

});

