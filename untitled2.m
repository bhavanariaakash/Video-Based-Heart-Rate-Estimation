clc; clear; close all;

%% --- Settings ---
duration = 20;   % seconds to record
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
cam = webcam;
cam.Resolution = '640x480';  % lower resolution for better FPS

disp(" Detecting face... Please stay still with good lighting.");
pause(2);

%% --- Detect Face Once---
frame = snapshot(cam);
frame = flip(frame, 2);
gray = rgb2gray(frame);
bboxes = step(faceDetector, gray);

if isempty(bboxes)
    clear cam;
    error(" No face detected. Try again with better lighting.");
end

[~, idx] = max(bboxes(:,3).*bboxes(:,4));
bbox = bboxes(idx,:);
x = bbox(1); y = bbox(2); w = bbox(3); h = bbox(4);

% Define smaller FOREHEAD ROI (top 15% of face box, centered)
roi_y1 = y;
roi_y2 = y + round(0.15*h);
roi_x1 = x + round(0.35*w);
roi_x2 = x + round(0.65*w);

disp(" Face detected. Recording... Stay still!");

%% --- Initialize Arrays ---
green_values = [];
timestamps = [];
start_time = tic;
frame_count = 0;
fps_window = 15;
live_bpm = NaN;

figure('Name','Heart Rate Detection','NumberTitle','off');


%% --- Recording Loop ---
frame_skip = 10;   % detect face every 10 frames
while toc(start_time) < duration
    frame = snapshot(cam);
    frame = flip(frame, 2);
    frame_count = frame_count + 1;

    % --- Detect face occasionally ---
    if mod(frame_count, frame_skip) == 1
        gray = rgb2gray(frame);
        bboxes = step(faceDetector, gray);

        if isempty(bboxes)
            frame = insertText(frame, [10 40], 'NO FACE DETECTED', ...
                               'FontSize', 16, 'BoxColor', 'red');
            imshow(frame); drawnow limitrate;
            continue;
        else
            [~, idx] = max(bboxes(:,3).*bboxes(:,4));
            bbox = bboxes(idx,:);
            x = bbox(1); y = bbox(2); w = bbox(3); h = bbox(4);
            roi_y1 = y;
            roi_y2 = y + round(0.15*h);
            roi_x1 = x + round(0.35*w);
            roi_x2 = x + round(0.65*w);
        end
    

    % --- Extract ROI and intensity ---
    roi = frame(roi_y1:roi_y2, roi_x1:roi_x2, :);
    green = mean(roi(:,:,2), 'all');
    green_values(end+1) = green;
    timestamps(end+1) = toc(start_time);

    % --- FPS Estimate ---
    if numel(timestamps) > 5
        fps_live = 1 / mean(diff(timestamps(max(1,end-fps_window):end)));
    else
        fps_live = NaN;
    end

    % --- Live Heart Rate (update every 15 frames) ---
    if numel(green_values) > 30 && mod(frame_count, 15) == 0
        fs = max(fps_live, 10); % prevent NaN
        low_cut = 1.0;  % 60 BPM
        high_cut = 2.2; % 132 BPM
        [b,a] = butter(3, [low_cut high_cut]/(fs/2), 'bandpass');
        filtered = filtfilt(b,a, green_values - mean(green_values));

        n = length(filtered);
        f = (0:n-1)*(fs/n);
        fft_val = abs(fft(filtered(1:end)));
        f = f(1:floor(n/2));
        fft_val = fft_val(1:floor(n/2));

        idx = (f >= low_cut & f <= high_cut);
        [~, pk] = max(fft_val(idx));
        freq_est = f(idx);
        live_bpm = freq_est(pk)*60;
    end

    % --- Display ---
    if mod(frame_count,5)==0
        frame = insertShape(frame,'Rectangle',[roi_x1 roi_y1 (roi_x2-roi_x1) (roi_y2-roi_y1)], ...
                            'Color','green','LineWidth',3);
        frame = insertText(frame,[10 10],sprintf("FPS: %.1f",fps_live),'FontSize',16,'BoxColor','yellow');
        if ~isnan(live_bpm)
            frame = insertText(frame,[10 40],sprintf("HR: %.0f BPM",live_bpm), ...
                               'FontSize',16,'BoxColor','red');
        end
        imshow(frame); drawnow limitrate;
    end
end


    if isempty(bboxes)
        % No face detected — show message and skip processing
        frame = insertText(frame,[10 40],'NO FACE DETECTED','FontSize',16,'BoxColor','red');
        imshow(frame); drawnow limitrate;
        continue;  % skip this iteration
    end

    % Proceed only if face detected
    [~, idx] = max(bboxes(:,3).*bboxes(:,4));  % largest face
    bbox = bboxes(idx,:);
    x = bbox(1); y = bbox(2); w = bbox(3); h = bbox(4);

    % Define ROI (forehead region)
    roi_y1 = y;
    roi_y2 = y + round(0.15*h);
    roi_x1 = x + round(0.35*w);
    roi_x2 = x + round(0.65*w);

    % Ensure ROI inside frame boundaries
    if roi_y2 > size(frame,1) || roi_x2 > size(frame,2)
        frame = insertText(frame,[10 40],'INVALID ROI','FontSize',16,'BoxColor','red');
        imshow(frame); drawnow limitrate;
        continue;
    end

    % Extract ROI and compute green intensity
    roi = frame(roi_y1:roi_y2, roi_x1:roi_x2, :);
    green = mean(roi(:,:,2), 'all');

    % Store values for HR calculation
    green_values(end+1) = green;
    timestamps(end+1) = toc(start_time);

    % Show ROI on frame
    frame = insertShape(frame,'Rectangle',[roi_x1 roi_y1 (roi_x2-roi_x1) (roi_y2-roi_y1)],'Color','green');
    imshow(frame);
    drawnow limitrate;


    %% --- FPS Calculation ---
    if frame_count > 5
        fps_live = 1 / mean(diff(timestamps(max(1,end-fps_window):end)));
    else
        fps_live = NaN;
    end

    %% --- Live Heart Rate Estimation (update every second) ---
    if numel(green_values) > 30 && mod(frame_count, 10) == 0
        fs = fps_live;
        if isnan(fs) || fs < 5
            continue;
        end
        % Restrict band to 1.25–2.0 Hz (≈ 75–120 BPM)
        low_cut = 1.25;
        high_cut = 2.0;
        if high_cut > fs/2
            high_cut = fs/2 - 0.1;
        end
        [b,a] = butter(3, [low_cut high_cut]/(fs/2), 'bandpass');
        filtered = filtfilt(b,a, green_values - mean(green_values));

        n = length(filtered);
        f = (0:n-1)*(fs/n);
        fft_val = abs(fft(filtered));
        f = f(1:floor(n/2));
        fft_val = fft_val(1:floor(n/2));

        idx = (f >= low_cut & f <= high_cut);
        [~, pk] = max(fft_val(idx));
        freq_est = f(idx);
        live_bpm = freq_est(pk)*60;
    end

    %% --- Visualization (every 5th frame) ---
    if mod(frame_count,5)==0
        frame = insertShape(frame,'Rectangle',[roi_x1 roi_y1 (roi_x2-roi_x1) (roi_y2-roi_y1)], ...
                            'Color','green','LineWidth',3);
        frame = insertText(frame,[10 10],sprintf("FPS: %.1f",fps_live),'FontSize',16,'BoxColor','yellow');
        if ~isnan(live_bpm)
            frame = insertText(frame,[10 40],sprintf("HR: %.0f BPM",live_bpm),'FontSize',16,'BoxColor','red');
        end
        imshow(frame); drawnow limitrate;
    end
end

clear cam;
disp("Recording complete.");

%% --- Post-processing for Final Heart Rate ---
fps = 1 / mean(diff(timestamps));
green_values = green_values - mean(green_values);

low_cut = 1.25;
high_cut = min(2.0, 0.45*fps);
[b,a] = butter(3, [low_cut high_cut]/(fps/2), 'bandpass');
filtered_signal = filtfilt(b,a, green_values);

n = length(filtered_signal);
frequencies = (0:n-1)*(fps/n);
fft_values = abs(fft(filtered_signal));
frequencies = frequencies(1:floor(n/2));
fft_values = fft_values(1:floor(n/2));

idx = (frequencies >= low_cut & frequencies <= high_cut);
filtered_freqs = frequencies(idx);
filtered_fft = fft_values(idx);
[~, peak_idx] = max(filtered_fft);
peak_freq = filtered_freqs(peak_idx);
bpm = peak_freq * 60;

fprintf('\n Measured FPS: %.2f\n', fps);
fprintf(' Final Estimated Heart Rate: %.2f BPM\n', bpm);

%% --- Plot ---
figure;
subplot(2,1,1);
plot(timestamps, filtered_signal, 'g');
title('Filtered Green Signal (Forehead ROI)');
xlabel('Time (s)'); ylabel('Intensity');

subplot(2,1,2);
plot(filtered_freqs*60, filtered_fft, 'r');
title('FFT (Heart Rate Detection)');
xlabel('Frequency (BPM)'); ylabel('Amplitude');
