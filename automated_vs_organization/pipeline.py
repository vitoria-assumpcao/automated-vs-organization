from automated_vs_organization.dicom.organize_dicom import organize_dicom
from automated_vs_organization.contours.organize_contours import organize_contours

from automated_vs_organization.images_and_labels.build_images_labels_full import (
    main as build_images_labels_full

)
from automated_vs_organization.masks.build_masks_full import (
    main as build_masks_full
)

from automated_vs_organization.training_set.build_training_set import main as build_training_set


def run_pipeline(
    tumor_only: bool = False,
):
    print("\n[1/5] Organizing DICOM files...")
    organize_dicom()

    print("\n[2/5] Organizing contour files...")
    organize_contours()

    if tumor_only:
        print("\n[3/5] Building tumor-only images and labels...")
        from automated_vs_organization.images_and_labels.build_images_labels_tumor import main
        main()

        print("\n[4/5] Building tumor-only masks...")
        from automated_vs_organization.masks.build_masks_tumor import main
        main()
    else:
        print("\n[3/5] Building full images and labels...")
        build_images_labels_full()

        print("\n[4/5] Building full masks...")
        build_masks_full()

    print("\n[5/5] Building final dataset split...")
    build_training_set()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline(tumor_only=True)   #Only tumor exams
    #run_pipeline(tumor_only=False) #Full exams

