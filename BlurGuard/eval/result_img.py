import os
import torch
import numpy as np
from collections import Counter, OrderedDict
import csv
import argparse


def order_counts(counts, file_names):
    
    ordered = OrderedDict((name, counts.get(name, 0)) for name in file_names)
    return ordered


def calculate_mean_std(values):
    
    filtered_values = [v for v in values if v is not None]  
    if len(filtered_values) == 0:
        return float('nan'), float('nan')  
    mean = np.mean(filtered_values)
    std = np.std(filtered_values)
    return mean, std

def metric_img(writer, base_path):
    
    try:
        sub_dirs = ['img']
        file_paths = [os.path.join(base_path, sub_dir, 'x_adv_metrics.bin') for sub_dir in sub_dirs]

        for sub_dir, file_path in zip(sub_dirs, file_paths):
            try:
                data = torch.load(file_path)
                
                ssim_mean, ssim_std = calculate_mean_std(data['ssim'])
                lpips_mean, lpips_std = calculate_mean_std(data['lpips'])
                psnr_mean, psnr_std = calculate_mean_std(data['psnr'])
                
                ia_mean, ia_std = calculate_mean_std(data.get('ia', []))
                adv_clip_mean, adv_clip_std = calculate_mean_std(data.get('adv_clip', []))
                
                if 'fid' in data and data['fid'] is not None:
                    fid = data['fid'][0] if isinstance(data['fid'], list) and len(data['fid']) > 0 else float('nan')
                else:
                    fid = float('nan')

                #print(f"{sub_dir} - LPIPS: {lpips_mean:.2f}, SSIM: {ssim_mean:.2f}, PSNR: {psnr_mean:.1f}, IA: {ia_mean:.2f}, FID: {fid:.2f}")
                print(f"Naturalness - LPIPS: {lpips_mean:.2f}, SSIM: {ssim_mean:.2f}, PSNR: {psnr_mean:.1f}, IA: {ia_mean:.2f}, FID: {fid:.2f}")

                writer.writerow([
                    "Naturalness",
                    f"${lpips_mean:.2f}±{lpips_std:.2f}$" if not np.isnan(lpips_mean) else "N/A",
                    f"${ssim_mean:.2f}±{ssim_std:.2f}$" if not np.isnan(ssim_mean) else "N/A",
                    f"${psnr_mean:.1f}±{psnr_std:.2f}$" if not np.isnan(psnr_mean) else "N/A",
                    f"${ia_mean:.2f}±{ia_std:.2f}$" if not np.isnan(ia_mean) else "N/A",
                    f"${adv_clip_mean:.2f}±{adv_clip_std:.2f}$" if not np.isnan(adv_clip_mean) else "N/A",
                    f"${fid:.2f}$" if not np.isnan(fid) else "N/A"
                ])
            except Exception as sub_dir_error:
                print(f"Error processing {sub_dir}: {sub_dir_error}")
    except Exception as e:
        print(f"Error processing files: {e}")

def metric_gen(writer, base_path):
    try:
        sub_dirs = ['gen/img', 'gen/jpeg', 'gen/jpeg_ups', 'gen/noise_ups', 'gen/diff_pure', 'gen/grid_pure', 'gen/impress', 'gen/pdm_pure']
        file_paths = [os.path.join(base_path, sub_dir, 'x_adv_metrics.bin') for sub_dir in sub_dirs]

        for sub_dir, file_path in zip(sub_dirs, file_paths):
            try:
                if not os.path.exists(file_path):
                    print(f".bin file not found: {file_path}")
                    continue
                    
                data = torch.load(file_path)
                
                ssim_mean, ssim_std = calculate_mean_std(data['ssim'])
                lpips_mean, lpips_std = calculate_mean_std(data['lpips'])
                psnr_mean, psnr_std = calculate_mean_std(data['psnr'])

                ia_mean, ia_std = calculate_mean_std(data['ia'])
                adv_clip_mean, adv_clip_std = calculate_mean_std(data['adv_clip'])
                
                fid = data['fid'][0] if 'fid' in data and data['fid'] else float('nan')
                
                writer.writerow([
                    sub_dir,
                    f"${lpips_mean:.2f}±{lpips_std:.2f}$" if not np.isnan(lpips_mean) else "N/A",
                    f"${ssim_mean:.2f}±{ssim_std:.2f}$" if not np.isnan(ssim_mean) else "N/A",
                    f"${psnr_mean:.1f}±{psnr_std:.2f}$" if not np.isnan(psnr_mean) else "N/A",
                    f"${ia_mean:.2f}±{ia_std:.2f}$" if not np.isnan(ia_mean) else "N/A",
                    f"${adv_clip_mean:.2f}±{adv_clip_std:.2f}$" if not np.isnan(adv_clip_mean) else "N/A",
                    f"${fid:.2f}$" if not np.isnan(fid) else "N/A"
                ])
            except Exception as sub_dir_error:
                print(f"Error processing {sub_dir}: {sub_dir_error}")
    except Exception as e:
        print(f"Error processing files: {e}")

def metric_worst(writer, base_path):
    try:
        sub_dirs = ['gen/img', 'gen/jpeg', 'gen/jpeg_ups', 'gen/noise_ups', 'gen/diff_pure', 'gen/grid_pure', 'gen/impress', 'gen/pdm_pure']
        file_paths = [os.path.join(base_path, sub_dir, 'x_adv_metrics.bin') for sub_dir in sub_dirs]

        metrics = {'ssim': [], 'lpips': [], 'psnr': [],'ia': [],'adv_clip': [],'fid':[]}

        valid_files = []
        for i, file_path in enumerate(file_paths):
            if os.path.exists(file_path):
                data = torch.load(file_path)
                for key in metrics.keys():
                    if key in data:
                        metrics[key].append(data[key])
                    else:
                        metrics[key].append([])
                valid_files.append(sub_dirs[i])

        if not valid_files:
            print("No valid .bin files found for worst calculation")
            return

        print("Metrics loaded:")
        for key in metrics.keys():
            print(f"{key}: {[len(row) for row in metrics[key]]} values per file")

        worst_ssim = []  
        worst_lpips = []  
        worst_psnr = []  
        worst_ia = []  
        worst_adv_clip = []  

        ssim_worst_indices = []
        lpips_worst_indices = []
        psnr_worst_indices = []
        ia_worst_indices = []
        adv_clip_worst_indices = []

        if len(metrics['ssim']) > 0 and len(metrics['ssim'][0]) > 0:
            num_samples = len(metrics['ssim'][0])  
            print(f"Total samples per metric: {num_samples}")

            for i in range(num_samples):  
                
                ssim_values = [metrics['ssim'][j][i] for j in range(len(valid_files)) if i < len(metrics['ssim'][j]) and metrics['ssim'][j][i] is not None]
                lpips_values = [metrics['lpips'][j][i] for j in range(len(valid_files)) if i < len(metrics['lpips'][j]) and metrics['lpips'][j][i] is not None]
                psnr_values = [metrics['psnr'][j][i] for j in range(len(valid_files)) if i < len(metrics['psnr'][j]) and metrics['psnr'][j][i] is not None]
                ia_values = [metrics['ia'][j][i] for j in range(len(valid_files)) if i < len(metrics['ia'][j]) and metrics['ia'][j][i] is not None]
                adv_clip_values = [metrics['adv_clip'][j][i] for j in range(len(valid_files)) if i < len(metrics['adv_clip'][j]) and metrics['adv_clip'][j][i] is not None]
                
                if ssim_values:
                    worst_ssim_value = max(ssim_values)
                    worst_ssim.append(worst_ssim_value)
                    ssim_worst_indices.append(ssim_values.index(worst_ssim_value))
                
                if lpips_values:
                    worst_lpips_value = min(lpips_values)
                    worst_lpips.append(worst_lpips_value)
                    lpips_worst_indices.append(lpips_values.index(worst_lpips_value))
                
                if psnr_values:
                    worst_psnr_value = max(psnr_values)
                    worst_psnr.append(worst_psnr_value)
                    psnr_worst_indices.append(psnr_values.index(worst_psnr_value))
                
                if ia_values:
                    worst_ia_value = max(ia_values)
                    worst_ia.append(worst_ia_value)
                    ia_worst_indices.append(ia_values.index(worst_ia_value))
                
                if adv_clip_values:
                    worst_adv_clip_value = max(adv_clip_values)
                    worst_adv_clip.append(worst_adv_clip_value)
                    adv_clip_worst_indices.append(adv_clip_values.index(worst_adv_clip_value))

            ssim_mean, ssim_std = calculate_mean_std(worst_ssim)
            lpips_mean, lpips_std = calculate_mean_std(worst_lpips)
            psnr_mean, psnr_std = calculate_mean_std(worst_psnr)
            ia_mean, ia_std = calculate_mean_std(worst_ia)
            adv_clip_mean, adv_clip_std = calculate_mean_std(worst_adv_clip)
            
            fid_values = metrics['fid']  
            flat_fid_values = [value for sublist in fid_values for value in sublist if value is not None]  
            min_fid = min(flat_fid_values) if flat_fid_values else float('nan')
            
            print("\nFinal Results:")
            print(f"SSIM (Worst-case Effectiveness): {ssim_mean:.2f} ± {ssim_std:.2f}")
            print(f"LPIPS (Worst-case Effectiveness): {lpips_mean:.2f} ± {lpips_std:.2f}")
            print(f"PSNR (Worst-case Effectiveness): {psnr_mean:.1f} ± {psnr_std:.2f}")
            print(f"i.a. (Worst-case Effectiveness): {ia_mean:.2f} ± {ia_std:.2f}")
            print(f"adv_clip (Worst-case Effectiveness): {adv_clip_mean:.2f} ± {adv_clip_std:.2f}")
            print(f"fid (Worst-case Effectiveness): {min_fid}")

            writer.writerow(['Worst-case Effectiveness',
                             f"${lpips_mean:.2f}±{lpips_std:.2f}$" if not np.isnan(lpips_mean) else "N/A",
                             f"${ssim_mean:.2f}±{ssim_std:.2f}$" if not np.isnan(ssim_mean) else "N/A",
                             f"${psnr_mean:.1f}±{psnr_std:.2f}$" if not np.isnan(psnr_mean) else "N/A",
                             f"${ia_mean:.2f}±{ia_std:.2f}$" if not np.isnan(ia_mean) else "N/A",
                             f"${adv_clip_mean:.2f}±{adv_clip_std:.2f}$" if not np.isnan(adv_clip_mean) else "N/A",
                             f"${min_fid:.2f}$" if not np.isnan(min_fid) else "N/A"])

    except Exception as e:
        print(f"Error processing files: {e}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Calculate metrics and save to CSV.")
    parser.add_argument("--path", required=True, help="Base path for metrics files.")
    args = parser.parse_args()

    base_path = args.path
    csv_file_path = os.path.join(base_path, 'results.csv')

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['subdir', 'LPIPS', 'SSIM', 'PSNR', 'IA', 'ADV_CLIP', 'FID'])
        
        metric_img(writer, base_path)
        metric_worst(writer, base_path)
        metric_gen(writer, base_path)

    print(f"\nResults saved to CSV file: {csv_file_path}")
