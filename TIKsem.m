clear all; close all; clc;

% ==== UČITAVANJE IRIS DATASETA ====
load fisheriris
rng(42); % Fiksiramo seed za reproduktivnost

% Koristi samo versicolor i virginica (kompleksniji problem)
fprintf('=== IRIS DATASET (versicolor i virginica) ===\n');
idx_hard = strcmp(species,'versicolor') | strcmp(species,'virginica');
X = meas(idx_hard,:);
species_subset = species(idx_hard);
species_numeric = grp2idx(species_subset); % 1=versicolor, 2=virginica

fprintf('Ukupno uzoraka: %d\n', size(X,1));
fprintf('Klasa 1 (versicolor): %d uzoraka\n', sum(species_numeric==1));
fprintf('Klasa 2 (virginica): %d uzoraka\n\n', sum(species_numeric==2));

num_classes = 2;
noise_level = 0.30;  % 30% šuma

% Dodavanje šuma i normalizacija
X_noisy = X + noise_level * randn(size(X)) .* std(X);
X_norm = (X_noisy - mean(X_noisy)) ./ std(X_noisy);

fprintf('Dodato %.0f%% Gaussian šuma\n', noise_level*100);

% One-hot encoding za mrežu
Y = zeros(size(X,1), num_classes);
for i = 1:size(X,1)
    Y(i, species_numeric(i)) = 1;
end
Y_discrete = species_numeric;

% Diskretizacija ulaza
num_bins = 10;
Xd = zeros(size(X_norm));
for i = 1:size(X_norm,2)
    edges = linspace(min(X_norm(:,i)), max(X_norm(:,i)), num_bins+1);
    Xd(:,i) = discretize(X_norm(:,i), edges);
    Xd(isnan(Xd(:,i)),i) = num_bins;
end

% ==== PODJELA NA TRAIN/VAL/TEST (60%/20%/20%) ====
train_ratio = 0.6;
val_ratio   = 0.2;
test_ratio  = 0.2;

N = size(X_norm,1);
train_idx = [];
val_idx = [];
test_idx = [];

% Stratifikacija po klasama
for class = 1:num_classes
    class_idx = find(species_numeric == class);
    n_class = length(class_idx);
    perm = randperm(n_class);
    n_train = round(train_ratio * n_class);
    n_val = round(val_ratio * n_class);
    
    train_idx = [train_idx; class_idx(perm(1:n_train))];
    val_idx = [val_idx; class_idx(perm(n_train+1:n_train+n_val))];
    test_idx = [test_idx; class_idx(perm(n_train+n_val+1:end))];
end

% Miješanje
train_idx = train_idx(randperm(length(train_idx)));
val_idx = val_idx(randperm(length(val_idx)));
test_idx = test_idx(randperm(length(test_idx)));

X_train = X_norm(train_idx,:);   Y_train = Y(train_idx,:);   Y_train_d = Y_discrete(train_idx);
X_val   = X_norm(val_idx,:);     Y_val   = Y(val_idx,:);     Y_val_d = Y_discrete(val_idx);
X_test  = X_norm(test_idx,:);    Y_test  = Y(test_idx,:);    Y_test_d = Y_discrete(test_idx);

Xd_train = Xd(train_idx,:);
Xd_val   = Xd(val_idx,:);
Xd_test  = Xd(test_idx,:);

fprintf('Podela: train=%d, val=%d, test=%d\n\n', length(Y_train_d), length(Y_val_d), length(Y_test_d));

% Referentna entropija
H_Y_true = compute_entropy(Y_train_d);
fprintf('=== REFERENTNE VREDNOSTI ===\n');
fprintf('H(Y_true) = %.4f bits (teorijski max = 1.0000)\n\n', H_Y_true);

% ==== TRI SCENARIJA ====
scenarios = {'Underfitting','Optimal','Overfitting'};
neurons_list = {[2], [6 6], [20 20 20]};
epoch_list = [20, 300, 2000];
lr = 0.05;

% Skladište rezultata za tabelarni prikaz
results = struct();

for s = 1:length(scenarios)
    fprintf('=========================================\n');
    fprintf('SCENARIO: %s\n', scenarios{s});
    fprintf('=========================================\n');

    neurons = neurons_list{s};
    epochs = epoch_list(s);

    net = patternnet(neurons);
    net.trainParam.epochs = epochs;
    net.trainParam.lr = lr;
    net.trainParam.showWindow = false;
    net.divideFcn = 'dividetrain';

    net = train(net, X_train', Y_train');

    % Predikcije
    Y_pred_train = net(X_train')';
    Y_pred_val   = net(X_val')';
    Y_pred_test  = net(X_test')';

    [~, Yp_train] = max(Y_pred_train, [], 2);
    [~, Yp_val]   = max(Y_pred_val, [], 2);
    [~, Yp_test]  = max(Y_pred_test, [], 2);

    % METRIKE
    H_Yp_train = compute_entropy(Yp_train);
    H_Yp_val   = compute_entropy(Yp_val);
    H_Yp_test  = compute_entropy(Yp_test);

    acc_train = mean(Yp_train == Y_train_d);
    acc_val   = mean(Yp_val == Y_val_d);
    acc_test  = mean(Yp_test == Y_test_d);

    % Međusobna informacija ulaz-izlaz
    I_XY = zeros(1,4);
    H_Y_given_X = zeros(1,4);
    for i = 1:4
        H_Y_given_X(i) = conditional_entropy(Xd_train(:,i), Yp_train);
        I_XY(i) = H_Yp_train - H_Y_given_X(i);
    end

    % Analiza neurona prvog sloja
    A1 = net.IW{1,1} * X_train' + net.b{1};
    A1 = 1./(1 + exp(-A1));
    
    A1d = zeros(size(A1'));
    for i = 1:size(A1,1)
        edges = linspace(min(A1(i,:)), max(A1(i,:)), num_bins+1);
        A1d(:,i) = discretize(A1(i,:)', edges);
        A1d(isnan(A1d(:,i)),i) = num_bins;
    end

    I_A1 = zeros(1, size(A1d,2));
    for i = 1:size(A1d,2)
        I_A1(i) = mutual_info(A1d(:,i), Yp_train);
    end

    % ISPIS REZULTATA
    fprintf('Arhitektura: [%s], Epohe: %d\n', num2str(neurons), epochs);

    fprintf('\n--- ACCURACY ---\n');
    fprintf('Train: %.2f%%  |  Val: %.2f%%  |  Test: %.2f%%\n', acc_train*100, acc_val*100, acc_test*100);
    
    fprintf('\n--- ENTROPIJA PREDIKCIJA H(Y) ---\n');
    fprintf('Train: %.4f  |  Val: %.4f  |  Test: %.4f\n', H_Yp_train, H_Yp_val, H_Yp_test);
    fprintf('GAP (val-train): %.4f\n', H_Yp_val - H_Yp_train);
    
    fprintf('\n--- USLOVNA ENTROPIJA H(Y|X_i) ---\n');
    fprintf('X1: %.4f  |  X2: %.4f  |  X3: %.4f  |  X4: %.4f\n', H_Y_given_X);
    
    fprintf('\n--- MEĐUSOBNA INFORMACIJA I(X_i ; Y) ---\n');
    fprintf('X1: %.4f  |  X2: %.4f  |  X3: %.4f  |  X4: %.4f\n', I_XY);
    fprintf('Prosjek: %.4f bits\n', mean(I_XY));
    
    fprintf('\n--- NEURONI SKRIVENOG SLOJA ---\n');
    fprintf('Ukupno neurona: %d\n', length(I_A1));
    fprintf('Prosječna I(A1; Y): %.4f bits\n', mean(I_A1));
    fprintf('Max I(A1; Y): %.4f  |  Min I(A1; Y): %.4f\n', max(I_A1), min(I_A1));
    
    useful = sum(I_A1 > 0.3);
    fprintf('Korisni neuroni (I>0.3): %d / %d (%.1f%%)\n\n', useful, length(I_A1), 100*useful/length(I_A1));

    fprintf('\n');
    
end

fprintf('\n=== ANALIZA ZAVRŠENA ===\n');

% ===== POMOĆNE FUNKCIJE =====

function H = compute_entropy(Y)
    Y = double(Y(:));
    classes = unique(Y);
    probs = zeros(length(classes),1);
    for i = 1:length(classes)
        probs(i) = sum(Y==classes(i)) / length(Y);
    end
    probs = probs(probs > 0);
    if isempty(probs)
        H = 0;
    else
        H = -sum(probs .* log2(probs));
    end
end

function H = conditional_entropy(X_col, Y_col)
    X_col = double(X_col(:));
    Y_col = double(Y_col(:));
    X_vals = unique(X_col);
    H = 0;
    for i = 1:length(X_vals)
        idx = (X_col == X_vals(i));
        p = sum(idx) / length(X_col);
        H = H + p * compute_entropy(Y_col(idx));
    end
end

function I = mutual_info(X_col, Y_col)
    I = compute_entropy(Y_col) - conditional_entropy(X_col, Y_col);
end