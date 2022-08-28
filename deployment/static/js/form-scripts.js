voices = '';


$(document).ready(function () {
    $('#ocr').click(function () {
        $('#working').show();
        var form_data = new FormData($('#ocr_form')[0]);
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function (data) {
                try {
                    $('#result').val(data)
                    $('#working').hide();
                }
                catch (err) {
                    $('#working').hide();
                }
                if (data.hasOwnProperty('error')) {
                    $('#working').hide();
                }
            },
        });
    });
});


$(document).on('change', '.btn-file :file', function () {
    var input = $(this),
        label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
    input.trigger('fileselect', [label]);
});


$('.btn-file :file').on('fileselect', function (event, label) {
    var input = $(this).parents('.input-group').find(':text'),
        log = label;
    if (input.length) {
        input.val(log);
    } else {
        if (log) alert(log);
    }
});
