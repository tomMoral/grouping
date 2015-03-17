%% Statistical difference

s = load('Reh_c3.txt');
[p, table, stats] = friedman(s)

[c, m, hp, gnames] = multcompare(stats, 'ctype', 'tukey-kramer', 'estimate', 'friedman', 'alpha', 0.05)