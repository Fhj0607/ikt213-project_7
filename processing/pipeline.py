from processing.preProcessing import (
    ToGrayscale,
    Sharpen,
    ApplyClahe,
    Brighten,
    Darken
)

from processing.analysis import (
    IsBlurry,
    IsLowContrast,
    IsTooDark,
    IsTooBright
)



def ProcessImage(image):

    processed = image.copy()

    report = []

    tooDark, brightnessScore = IsTooDark(processed)

    if tooDark:
        processed = Brighten(processed)
        report.append(f"Brightened image | brightness score: {brightnessScore:.2f}")

    else:
        tooBright, brightnessScore = IsTooBright(processed)

        if tooBright:
            processed = Darken(processed)
            report.append(f"Darkened image | brightness score: {brightnessScore:.2f}")

    processed = ToGrayscale(processed)

    blurry, blurScore = IsBlurry(processed)

    if blurry:
        processed = Sharpen(processed)
        report.append(f"Sharpened image | blur score: {blurScore:.2f}")

    lowContrast, contrastScore = IsLowContrast(processed)

    if lowContrast:
        processed = ApplyClahe(processed)
        report.append(f"Applied CLAHE | contrast score: {contrastScore:.2f}")

    return processed, report