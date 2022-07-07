import glob
import os
import torch
import numpy as np


def classify_ivd(classificationNet, ivds, device):
    # Classify
    pred_pf = torch.Tensor().to(device).long()
    pred_nar = torch.Tensor().to(device).long()
    pred_ccs = torch.Tensor().to(device).long()
    pred_spon = torch.Tensor().to(device).long()
    pred_ued = torch.Tensor().to(device).long()
    pred_led = torch.Tensor().to(device).long()
    pred_umc = torch.Tensor().to(device).long()
    pred_lmc = torch.Tensor().to(device).long()
    with torch.no_grad():
        for image in ivds:
            # Augmentations
            num_rows, num_cols, num_slices = image.shape
            max_cols = num_cols - 48
            min_cols = 48
            max_rows = num_rows - 40
            min_rows = 40

            cum_image = arr = np.zeros((1, 1, 9, 112, 224), np.float32)
            for col_s in range(-16, 16 + 1, 16):
                for row_s in range(-16, 16 + 1, 16):
                    # 112 x 224 x 9 Slices
                    image1_1 = image[
                        min_rows + row_s : max_rows + row_s,
                        min_cols + col_s : max_cols + col_s,
                        2:11,
                    ]
                    image2_1 = image[
                        min_rows + row_s : max_rows + row_s,
                        min_cols + col_s : max_cols + col_s,
                        3:12,
                    ]
                    image3_1 = image[
                        min_rows + row_s : max_rows + row_s,
                        min_cols + col_s : max_cols + col_s,
                        4:13,
                    ]
                    image1_2 = np.flip(image1_1, axis=2).copy()
                    image2_2 = np.flip(image2_1, axis=2).copy()
                    image3_2 = np.flip(image3_1, axis=2).copy()

                    # Ready the images
                    image1_1 = (
                        np.transpose(image1_1, (2, 0, 1))[None, None, :, :, :] - 0.2386
                    )  # Channel mean = 0.2386
                    image2_1 = (
                        np.transpose(image2_1, (2, 0, 1))[None, None, :, :, :] - 0.2386
                    )  # Channel mean = 0.2386
                    image3_1 = (
                        np.transpose(image3_1, (2, 0, 1))[None, None, :, :, :] - 0.2386
                    )  # Channel mean = 0.2386
                    image1_2 = (
                        np.transpose(image1_2, (2, 0, 1))[None, None, :, :, :] - 0.2386
                    )  # Channel mean = 0.2386
                    image2_2 = (
                        np.transpose(image2_2, (2, 0, 1))[None, None, :, :, :] - 0.2386
                    )  # Channel mean = 0.2386
                    image3_2 = (
                        np.transpose(image3_2, (2, 0, 1))[None, None, :, :, :] - 0.2386
                    )  # Channel mean = 0.2386
                    cum_image = np.concatenate(
                        (
                            cum_image,
                            image1_1,
                            image2_1,
                            image3_1,
                            image1_2,
                            image2_2,
                            image3_2,
                        ),
                        0,
                    )

            image = np.delete(cum_image, 0, axis=0)
            image = torch.tensor(image).to(device)

            # forward
            net_output = classificationNet(image.float())

            # accumulate prediction and labels
            _, predicted_pf = torch.sum(net_output[0], 0).max(0)
            _, predicted_nar = torch.sum(net_output[1], 0).max(0)
            _, predicted_ccs = torch.sum(net_output[2], 0).max(0)
            _, predicted_spon = torch.sum(net_output[3], 0).max(0)
            _, predicted_ued = torch.sum(net_output[4], 0).max(0)
            _, predicted_led = torch.sum(net_output[5], 0).max(0)
            _, predicted_umc = torch.sum(net_output[6], 0).max(0)
            _, predicted_lmc = torch.sum(net_output[7], 0).max(0)
            pred_pf = torch.cat((pred_pf, predicted_pf[None]), 0)
            pred_nar = torch.cat((pred_nar, predicted_nar[None]), 0)
            pred_ccs = torch.cat((pred_ccs, predicted_ccs[None]), 0)
            pred_spon = torch.cat((pred_spon, predicted_spon[None]), 0)
            pred_ued = torch.cat((pred_ued, predicted_ued[None]), 0)
            pred_led = torch.cat((pred_led, predicted_led[None]), 0)
            pred_umc = torch.cat((pred_umc, predicted_umc[None]), 0)
            pred_lmc = torch.cat((pred_lmc, predicted_lmc[None]), 0)

    pred_pf = pred_pf.cpu().numpy()
    pred_nar = pred_nar.cpu().numpy()
    pred_ccs = pred_ccs.cpu().numpy()
    pred_spon = pred_spon.cpu().numpy()
    pred_ued = pred_ued.cpu().numpy()
    pred_led = pred_led.cpu().numpy()
    pred_umc = pred_umc.cpu().numpy()
    pred_lmc = pred_lmc.cpu().numpy()

    gradings = {}
    gradings["Pfirrmann"] = pred_pf
    gradings["Narrowing"] = pred_nar
    gradings["CentralCanalStenosis"] = pred_ccs
    gradings["Spondylolisthesis"] = pred_spon
    gradings["UpperEndplateDefect"] = pred_ued
    gradings["LowerEndplateDefect"] = pred_led
    gradings["UpperMarrow"] = pred_umc
    gradings["LowerMarrow"] = pred_lmc
    return gradings


def format_volume_for_classification_net(ivd_volume):
    num_rows, num_cols, num_slices = ivd_volume.shape
    max_cols = num_cols - 48
    min_cols = 48
    max_rows = num_rows - 40
    min_rows = 40
    image = ivd_volume[min_rows:max_rows, min_cols:max_cols, 3:12]
    image = torch.tensor(np.transpose(image, (2, 0, 1))[None, None, :, :, :])
    return image


def classify_ivd_v2_resnet(classificationNet, ivds, device):
    pred_pf = torch.Tensor().to(device).long()
    pred_nar = torch.Tensor().to(device).long()
    pred_ccs = torch.Tensor().to(device).long()
    pred_spon = torch.Tensor().to(device).long()
    pred_ued = torch.Tensor().to(device).long()
    pred_led = torch.Tensor().to(device).long()
    pred_umc = torch.Tensor().to(device).long()
    pred_lmc = torch.Tensor().to(device).long()
    pred_fsl = torch.Tensor().to(device).long()
    pred_fsr = torch.Tensor().to(device).long()
    pred_hrn = torch.Tensor().to(device).long()
    for ivd in ivds:
        with torch.no_grad():
            # Augmentations
            image = torch.tensor(ivd)[None, None, :, :, :].float().to(device)
            # forward
            net_output = classificationNet(image)
            # accumulate prediction and labels
            _, predicted_pf = net_output[0].squeeze().max(0)
            _, predicted_nar = net_output[1].squeeze().max(0)
            _, predicted_ccs = net_output[2].squeeze().max(0)
            _, predicted_spon = net_output[3].squeeze().max(0)
            _, predicted_ued = net_output[4].squeeze().max(0)
            _, predicted_led = net_output[5].squeeze().max(0)
            _, predicted_umc = net_output[6].squeeze().max(0)
            _, predicted_lmc = net_output[7].squeeze().max(0)
            _, predicted_fsl = net_output[8].squeeze().max(0)
            _, predicted_fsr = net_output[9].squeeze().max(0)
            _, predicted_hrn = net_output[10].squeeze().max(0)
            pred_pf = torch.cat((pred_pf, predicted_pf[None]), 0)
            pred_nar = torch.cat((pred_nar, predicted_nar[None]), 0)
            pred_ccs = torch.cat((pred_ccs, predicted_ccs[None]), 0)
            pred_spon = torch.cat((pred_spon, predicted_spon[None]), 0)
            pred_ued = torch.cat((pred_ued, predicted_ued[None]), 0)
            pred_led = torch.cat((pred_led, predicted_led[None]), 0)
            pred_umc = torch.cat((pred_umc, predicted_umc[None]), 0)
            pred_lmc = torch.cat((pred_lmc, predicted_lmc[None]), 0)
            pred_fsl = torch.cat((pred_fsl, predicted_fsl[None]), 0)
            pred_fsr = torch.cat((pred_fsr, predicted_fsr[None]), 0)
            pred_hrn = torch.cat((pred_hrn, predicted_hrn[None]), 0)
            torch.cuda.empty_cache()

    pred_pf = pred_pf.cpu().numpy()
    pred_nar = pred_nar.cpu().numpy()
    pred_ccs = pred_ccs.cpu().numpy()
    pred_spon = pred_spon.cpu().numpy()
    pred_ued = pred_ued.cpu().numpy()
    pred_led = pred_led.cpu().numpy()
    pred_umc = pred_umc.cpu().numpy()
    pred_lmc = pred_lmc.cpu().numpy()
    pred_fsl = pred_fsl.cpu().numpy()
    pred_fsr = pred_fsr.cpu().numpy()
    pred_hrn = pred_hrn.cpu().numpy()

    gradings = {}
    gradings["Pfirrmann"] = pred_pf
    gradings["Narrowing"] = pred_nar
    gradings["CentralCanalStenosis"] = pred_ccs
    gradings["Spondylolisthesis"] = pred_spon
    gradings["UpperEndplateDefect"] = pred_ued
    gradings["LowerEndplateDefect"] = pred_led
    gradings["UpperMarrow"] = pred_umc
    gradings["LowerMarrow"] = pred_lmc
    gradings["ForaminalStenosisLeft"] = pred_fsl
    gradings["ForaminalStenosisRight"] = pred_fsr
    gradings["Herniation"] = pred_hrn
    return gradings


def classify_ivd_no_aug(classificationNet, ivds, device):
    # Classify
    pred_pf = torch.Tensor().to(device).long()
    pred_nar = torch.Tensor().to(device).long()
    pred_ccs = torch.Tensor().to(device).long()
    pred_spon = torch.Tensor().to(device).long()
    pred_ued = torch.Tensor().to(device).long()
    pred_led = torch.Tensor().to(device).long()
    pred_umc = torch.Tensor().to(device).long()
    pred_lmc = torch.Tensor().to(device).long()
    with torch.no_grad():
        for image in ivds:
            # Augmentations
            num_rows, num_cols, num_slices = image.shape
            min_cols = 47
            max_cols = 271
            min_rows = 39
            max_rows = 151

            image = image[min_rows:max_rows, min_cols:max_cols, 3:12] - 0.2386
            image = torch.tensor(
                np.transpose(image, (2, 0, 1))[None, None, :, :, :]
            ).to(device)

            # forward
            net_output = classificationNet(image.float())

            # accumulate prediction and labels
            _, predicted_pf = net_output[0].squeeze().max(0)
            _, predicted_nar = net_output[1].squeeze().max(0)
            _, predicted_ccs = net_output[2].squeeze().max(0)
            _, predicted_spon = net_output[3].squeeze().max(0)
            _, predicted_ued = net_output[4].squeeze().max(0)
            _, predicted_led = net_output[5].squeeze().max(0)
            _, predicted_umc = net_output[6].squeeze().max(0)
            _, predicted_lmc = net_output[7].squeeze().max(0)
            pred_pf = torch.cat((pred_pf, predicted_pf[None]), 0)
            pred_nar = torch.cat((pred_nar, predicted_nar[None]), 0)
            pred_ccs = torch.cat((pred_ccs, predicted_ccs[None]), 0)
            pred_spon = torch.cat((pred_spon, predicted_spon[None]), 0)
            pred_ued = torch.cat((pred_ued, predicted_ued[None]), 0)
            pred_led = torch.cat((pred_led, predicted_led[None]), 0)
            pred_umc = torch.cat((pred_umc, predicted_umc[None]), 0)
            pred_lmc = torch.cat((pred_lmc, predicted_lmc[None]), 0)

    pred_pf = pred_pf.cpu().numpy()
    pred_nar = pred_nar.cpu().numpy()
    pred_ccs = pred_ccs.cpu().numpy()
    pred_spon = pred_spon.cpu().numpy()
    pred_ued = pred_ued.cpu().numpy()
    pred_led = pred_led.cpu().numpy()
    pred_umc = pred_umc.cpu().numpy()
    pred_lmc = pred_lmc.cpu().numpy()

    gradings = {}
    gradings["Pfirrmann"] = pred_pf
    gradings["Narrowing"] = pred_nar
    gradings["CentralCanalStenosis"] = pred_ccs
    gradings["Spondylolisthesis"] = pred_spon
    gradings["UpperEndplateDefect"] = pred_ued
    gradings["LowerEndplateDefect"] = pred_led
    gradings["UpperMarrow"] = pred_umc
    gradings["LowerMarrow"] = pred_lmc
    return gradings


def classify_ivd_no_aug_spinenetV1(classificationNet, ivds, device):
    # Classify
    pred_pf = torch.Tensor().to(device).long()
    pred_nar = torch.Tensor().to(device).long()
    pred_ccs = torch.Tensor().to(device).long()
    pred_spon = torch.Tensor().to(device).long()
    pred_ued = torch.Tensor().to(device).long()
    pred_led = torch.Tensor().to(device).long()
    pred_umc = torch.Tensor().to(device).long()
    pred_lmc = torch.Tensor().to(device).long()
    with torch.no_grad():
        for image in ivds:
            # Augmentations
            num_rows, num_cols, num_slices = image.shape
            min_cols = 47
            max_cols = 271
            min_rows = 39
            max_rows = 151

            # image = image[min_rows:max_rows,min_cols:max_cols,3:12] - 0.2386
            # SpineNet V1 normalization
            image = image[min_rows:max_rows, min_cols:max_cols, 3:12]
            image = torch.tensor(
                np.transpose(image, (2, 0, 1))[None, None, :, :, :]
            ).to(device)

            # forward
            net_output = classificationNet(image.float())

            # accumulate prediction and labels
            _, predicted_pf = net_output[0].squeeze().max(0)
            _, predicted_nar = net_output[1].squeeze().max(0)
            _, predicted_ccs = net_output[2].squeeze().max(0)
            _, predicted_spon = net_output[3].squeeze().max(0)
            _, predicted_ued = net_output[4].squeeze().max(0)
            _, predicted_led = net_output[5].squeeze().max(0)
            _, predicted_umc = net_output[6].squeeze().max(0)
            _, predicted_lmc = net_output[7].squeeze().max(0)
            pred_pf = torch.cat((pred_pf, predicted_pf[None]), 0)
            pred_nar = torch.cat((pred_nar, predicted_nar[None]), 0)
            pred_ccs = torch.cat((pred_ccs, predicted_ccs[None]), 0)
            pred_spon = torch.cat((pred_spon, predicted_spon[None]), 0)
            pred_ued = torch.cat((pred_ued, predicted_ued[None]), 0)
            pred_led = torch.cat((pred_led, predicted_led[None]), 0)
            pred_umc = torch.cat((pred_umc, predicted_umc[None]), 0)
            pred_lmc = torch.cat((pred_lmc, predicted_lmc[None]), 0)

    pred_pf = pred_pf.cpu().numpy()
    pred_nar = pred_nar.cpu().numpy()
    pred_ccs = pred_ccs.cpu().numpy()
    pred_spon = pred_spon.cpu().numpy()
    pred_ued = pred_ued.cpu().numpy()
    pred_led = pred_led.cpu().numpy()
    pred_umc = pred_umc.cpu().numpy()
    pred_lmc = pred_lmc.cpu().numpy()

    gradings = {}
    gradings["Pfirrmann"] = pred_pf
    gradings["Narrowing"] = pred_nar
    gradings["CentralCanalStenosis"] = pred_ccs
    gradings["Spondylolisthesis"] = pred_spon
    gradings["UpperEndplateDefect"] = pred_ued
    gradings["LowerEndplateDefect"] = pred_led
    gradings["UpperMarrow"] = pred_umc
    gradings["LowerMarrow"] = pred_lmc
    return gradings


def classify_ivd_spinenetV1(classificationNet, ivds, device):
    # Classify
    pred_pf = torch.Tensor().to(device).long()
    pred_nar = torch.Tensor().to(device).long()
    pred_ccs = torch.Tensor().to(device).long()
    pred_spon = torch.Tensor().to(device).long()
    pred_ued = torch.Tensor().to(device).long()
    pred_led = torch.Tensor().to(device).long()
    pred_umc = torch.Tensor().to(device).long()
    pred_lmc = torch.Tensor().to(device).long()
    with torch.no_grad():
        for image in ivds:
            # Augmentations
            num_rows, num_cols, num_slices = image.shape
            max_cols = num_cols - 48
            min_cols = 48
            max_rows = num_rows - 40
            min_rows = 40

            cum_image = arr = np.zeros((1, 1, 9, 112, 224), np.float32)
            for col_s in range(-16, 16 + 1, 16):
                for row_s in range(-16, 16 + 1, 16):
                    # 112 x 224 x 9 Slices
                    image1_1 = image[
                        min_rows + row_s : max_rows + row_s,
                        min_cols + col_s : max_cols + col_s,
                        2:11,
                    ]
                    image2_1 = image[
                        min_rows + row_s : max_rows + row_s,
                        min_cols + col_s : max_cols + col_s,
                        3:12,
                    ]
                    image3_1 = image[
                        min_rows + row_s : max_rows + row_s,
                        min_cols + col_s : max_cols + col_s,
                        4:13,
                    ]
                    image1_2 = np.flip(image1_1, axis=2).copy()
                    image2_2 = np.flip(image2_1, axis=2).copy()
                    image3_2 = np.flip(image3_1, axis=2).copy()

                    # Ready the images
                    image1_1 = np.transpose(image1_1, (2, 0, 1))[
                        None, None, :, :, :
                    ]  # Channel mean = 0.2386
                    image2_1 = np.transpose(image2_1, (2, 0, 1))[
                        None, None, :, :, :
                    ]  # Channel mean = 0.2386
                    image3_1 = np.transpose(image3_1, (2, 0, 1))[
                        None, None, :, :, :
                    ]  # Channel mean = 0.2386
                    image1_2 = np.transpose(image1_2, (2, 0, 1))[
                        None, None, :, :, :
                    ]  # Channel mean = 0.2386
                    image2_2 = np.transpose(image2_2, (2, 0, 1))[
                        None, None, :, :, :
                    ]  # Channel mean = 0.2386
                    image3_2 = np.transpose(image3_2, (2, 0, 1))[
                        None, None, :, :, :
                    ]  # Channel mean = 0.2386
                    cum_image = np.concatenate(
                        (
                            cum_image,
                            image1_1,
                            image2_1,
                            image3_1,
                            image1_2,
                            image2_2,
                            image3_2,
                        ),
                        0,
                    )

            image = np.delete(cum_image, 0, axis=0)
            # SpineNet V1 normalization
            image[:, :, 0, :, :] -= 0.1351
            image[:, :, 1, :, :] -= 0.1363
            image[:, :, 2, :, :] -= 0.1371
            image[:, :, 3, :, :] -= 0.1358
            image[:, :, 4, :, :] -= 0.1348
            image[:, :, 5, :, :] -= 0.1365
            image[:, :, 6, :, :] -= 0.1368
            image[:, :, 7, :, :] -= 0.1364
            image[:, :, 8, :, :] -= 0.1357
            image = torch.tensor(image).to(device)

            # forward
            net_output = classificationNet(image.float())

            # accumulate prediction and labels
            _, predicted_pf = torch.sum(net_output[0], 0).max(0)
            _, predicted_nar = torch.sum(net_output[1], 0).max(0)
            _, predicted_ccs = torch.sum(net_output[2], 0).max(0)
            _, predicted_spon = torch.sum(net_output[3], 0).max(0)
            _, predicted_ued = torch.sum(net_output[4], 0).max(0)
            _, predicted_led = torch.sum(net_output[5], 0).max(0)
            _, predicted_umc = torch.sum(net_output[6], 0).max(0)
            _, predicted_lmc = torch.sum(net_output[7], 0).max(0)
            pred_pf = torch.cat((pred_pf, predicted_pf[None]), 0)
            pred_nar = torch.cat((pred_nar, predicted_nar[None]), 0)
            pred_ccs = torch.cat((pred_ccs, predicted_ccs[None]), 0)
            pred_spon = torch.cat((pred_spon, predicted_spon[None]), 0)
            pred_ued = torch.cat((pred_ued, predicted_ued[None]), 0)
            pred_led = torch.cat((pred_led, predicted_led[None]), 0)
            pred_umc = torch.cat((pred_umc, predicted_umc[None]), 0)
            pred_lmc = torch.cat((pred_lmc, predicted_lmc[None]), 0)

    pred_pf = pred_pf.cpu().numpy()
    pred_nar = pred_nar.cpu().numpy()
    pred_ccs = pred_ccs.cpu().numpy()
    pred_spon = pred_spon.cpu().numpy()
    pred_ued = pred_ued.cpu().numpy()
    pred_led = pred_led.cpu().numpy()
    pred_umc = pred_umc.cpu().numpy()
    pred_lmc = pred_lmc.cpu().numpy()

    gradings = {}
    gradings["Pfirrmann"] = pred_pf
    gradings["Narrowing"] = pred_nar
    gradings["CentralCanalStenosis"] = pred_ccs
    gradings["Spondylolisthesis"] = pred_spon
    gradings["UpperEndplateDefect"] = pred_ued
    gradings["LowerEndplateDefect"] = pred_led
    gradings["UpperMarrow"] = pred_umc
    gradings["LowerMarrow"] = pred_lmc
    return gradings
