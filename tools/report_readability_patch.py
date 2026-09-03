from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"Expected block not found in {path}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Experiment 14A: add a visual Pareto summary and promote the remaining raw
# tensor-activity tabular block to a numbered table.
# ---------------------------------------------------------------------------
replace_once(
    "report/sections/08_experiment_14a_fmnist_binary.tex",
    r"""\end{table}

The selected event point strictly improves the tested EF Pareto frontier.""",
    r"""\end{table}

Figure~\ref{fig:exp14a-pareto} visualizes the same operating points on a logarithmic communication axis and makes the communication--performance separation explicit.

\ExpFourteenAParetoFigure

The selected event point strictly improves the tested EF Pareto frontier.""",
)

replace_once(
    "report/sections/08_experiment_14a_fmnist_binary.tex",
    r"""The fraction of parameters that fire at least once is approximately $65.1\%$. This is lower than in Experiment~13B and is mainly caused by the large first-layer matrix. Tensor-wise activity is
\begin{center}
\begin{tabular}{lrr}
\toprule
Tensor & Events/parameter & Never-fired fraction\\
\midrule
$W_1$ & 17.49 & 0.355\\
$b_1$ & 39.16 & 0.156\\
$W_2$ & 36.25 & 0.072\\
$b_2$ & 122.56 & 0\\
$W_3$ & 134.81 & 0\\
$b_3$ & 573.0 & 0\\
\bottomrule
\end{tabular}
\end{center}
There is therefore no evidence of a dead upper layer:""",
    r"""The fraction of parameters that fire at least once is approximately $65.1\%$. This is lower than in Experiment~13B and is mainly caused by the large first-layer matrix. Table~\ref{tab:exp14a-tensor-activity} resolves the activity by tensor.
\begin{table}[ht]
\centering
\small
\caption{Experiment~14A event activity by parameter tensor for the selected strong-partition run.}
\label{tab:exp14a-tensor-activity}
\begin{tabular}{lrr}
\toprule
Tensor & Events/parameter & Never-fired fraction\\
\midrule
$W_1$ & 17.49 & 0.355\\
$b_1$ & 39.16 & 0.156\\
$W_2$ & 36.25 & 0.072\\
$b_2$ & 122.56 & 0\\
$W_3$ & 134.81 & 0\\
$b_3$ & 573.0 & 0\\
\bottomrule
\end{tabular}
\end{table}
There is therefore no evidence of a dead upper layer:""",
)

# ---------------------------------------------------------------------------
# Experiment 14B: replace the long one-line boxed result with a wrapping prose
# callout. The visual seed summary lives in the companion 08bc section.
# ---------------------------------------------------------------------------
replace_once(
    "report/sections/08b_experiment_14b_fmnist_multiclass.tex",
    r"""For the default strong partition with seed $2400$, the selected event model reaches
\begin{equation}
\boxed{
\text{test CE}=0.64007,\quad
A=78.23\%,\quad
A_{\min}=35.3\%,\quad
B=34.75\ \text{Mbit}.
}
\end{equation}
The matched references are summarized in Table~\ref{tab:exp14b-primary}.""",
    r"""For the default strong partition with seed $2400$, the selected event model reaches the following representative operating point.
\begin{takeawaybox}[title={Experiment 14B primary event result}]
Test CE is $0.64007$, overall accuracy is $78.23\%$, worst-class accuracy is $35.3\%$, and logical uplink payload is $34.75$~Mbit.
\end{takeawaybox}
The matched references are summarized in Table~\ref{tab:exp14b-primary}.""",
)

# ---------------------------------------------------------------------------
# Experiment 14C: number the three result tables that were raw center/tabular
# blocks, reference them in prose, add the matched-work figure, and replace the
# final overwide mechanism box with a wrapping takeaway.
# ---------------------------------------------------------------------------
replace_once(
    "report/sections/08d_experiment_14c_event_share_fairness.tex",
    r"""The Spearman correlations between output event share and final class accuracy are approximately
\begin{center}
\begin{tabular}{lr}
\toprule
Scenario & $\rho(s_c^{\mathrm{out}},A_c)$\\
\midrule
IID, equal periods & $-0.782$\\
IID, heterogeneous periods & $-0.758$\\
Strong, equal periods & $-0.745$\\
Strong, heterogeneous identity & $-0.588$\\
Strong, heterogeneous reverse & $-0.285$\\
Strong, heterogeneous mixed & $-0.298$\\
Strong seed 2500, equal periods & $-0.851$\\
Strong seed 2500, heterogeneous periods & $-0.333$\\
\bottomrule
\end{tabular}
\end{center}""",
    r"""Table~\ref{tab:exp14c-output-share-correlation} reports the Spearman correlations between output event share and final class accuracy.
\begin{table}[ht]
\centering
\small
\caption{Experiment~14C correlation between class-associated output-event share and final class accuracy.}
\label{tab:exp14c-output-share-correlation}
\begin{tabular}{lr}
\toprule
Scenario & $\rho(s_c^{\mathrm{out}},A_c)$\\
\midrule
IID, equal periods & $-0.782$\\
IID, heterogeneous periods & $-0.758$\\
Strong, equal periods & $-0.745$\\
Strong, heterogeneous identity & $-0.588$\\
Strong, heterogeneous reverse & $-0.285$\\
Strong, heterogeneous mixed & $-0.298$\\
Strong seed 2500, equal periods & $-0.851$\\
Strong seed 2500, heterogeneous periods & $-0.333$\\
\bottomrule
\end{tabular}
\end{table}""",
)

replace_once(
    "report/sections/08d_experiment_14c_event_share_fairness.tex",
    r"""Strong-skew runs were repeated while permuting which semantic class is dominant on each compute slot. At the same $650$-tick horizon, the event results within one fixed numerical environment are
\begin{center}
\begin{tabular}{lrrr}
\toprule
Assignment & Test CE & Accuracy & Worst class\\
\midrule
Identity & 0.6322 & 0.7769 & 0.302\\
Reverse & 0.6521 & 0.7719 & \textbf{0.537}\\
Mixed & 0.6686 & 0.7524 & 0.264\\
\bottomrule
\end{tabular}
\end{center}""",
    r"""Strong-skew runs were repeated while permuting which semantic class is dominant on each compute slot. Table~\ref{tab:exp14c-period-assignment} reports the event results at the same $650$-tick horizon within one fixed numerical environment.
\begin{table}[ht]
\centering
\small
\caption{Experiment~14C sensitivity to the assignment between semantic class and client compute period.}
\label{tab:exp14c-period-assignment}
\begin{tabular}{lrrr}
\toprule
Assignment & Test CE & Accuracy & Worst class\\
\midrule
Identity & 0.6322 & 0.7769 & 0.302\\
Reverse & 0.6521 & 0.7719 & \textbf{0.537}\\
Mixed & 0.6686 & 0.7524 & 0.264\\
\bottomrule
\end{tabular}
\end{table}""",
)

replace_once(
    "report/sections/08d_experiment_14c_event_share_fairness.tex",
    r"""The matched-work event results are
\begin{center}
\begin{tabular}{llrrr}
\toprule
Regime & Compute & Test CE & Accuracy & Worst class\\
\midrule
IID, seed 2400 & equal, matched & 0.5389 & 0.8088 & \textbf{0.528}\\
                 & heterogeneous & 0.6721 & 0.7391 & 0.300\\
Strong, seed 2400 & equal, matched & 0.5675 & 0.8045 & \textbf{0.443}\\
                    & heterogeneous & 0.6322 & 0.7769 & 0.302\\
Strong, seed 2500 & equal, matched & 0.5860 & 0.8034 & \textbf{0.508}\\
                    & heterogeneous & 0.7491 & 0.7303 & 0.239\\
\bottomrule
\end{tabular}
\end{center}

This resolves an ambiguity""",
    r"""The matched-work event results are summarized in Table~\ref{tab:exp14c-matched-work}.
\begin{table}[ht]
\centering
\small
\caption{Experiment~14C event results with equal and heterogeneous compute schedules after matching the total amount of local work.}
\label{tab:exp14c-matched-work}
\begin{tabular}{llrrr}
\toprule
Regime & Compute & Test CE & Accuracy & Worst class\\
\midrule
IID, seed 2400 & equal, matched & 0.5389 & 0.8088 & \textbf{0.528}\\
                 & heterogeneous & 0.6721 & 0.7391 & 0.300\\
Strong, seed 2400 & equal, matched & 0.5675 & 0.8045 & \textbf{0.443}\\
                    & heterogeneous & 0.6322 & 0.7769 & 0.302\\
Strong, seed 2500 & equal, matched & 0.5860 & 0.8034 & \textbf{0.508}\\
                    & heterogeneous & 0.7491 & 0.7303 & 0.239\\
\bottomrule
\end{tabular}
\end{table}

Figure~\ref{fig:exp14c-matched-work} isolates the worst-class effect and shows that the degradation is large in all three matched-work comparisons.

\ExpFourteenCMatchedWorkFigure

This resolves an ambiguity""",
)

replace_once(
    "report/sections/08d_experiment_14c_event_share_fairness.tex",
    r"""The resulting mechanism picture is more precise than the initial fairness hypothesis:
\begin{equation}
\boxed{
\text{compute heterogeneity}
\;\longrightarrow\;
\begin{matrix}
\text{general asynchronous optimization delay}\\
+\\
\text{event-specific nonlinear traffic reshaping}
\end{matrix}
\;\longrightarrow\;
\text{partition-sensitive class transients}.
}
\end{equation}""",
    r"""The resulting mechanism picture is more precise than the initial fairness hypothesis.
\begin{takeawaybox}[title={Experiment 14C mechanism summary}]
Compute heterogeneity produces a general asynchronous optimization delay and, for the event method, an additional nonlinear reshaping of communication traffic. Their interaction produces partition-sensitive class transients; the data do not support a simple event-starvation explanation.
\end{takeawaybox}""",
)

# ---------------------------------------------------------------------------
# Experiment 15A: promote all report-facing raw tables to floats, add Pareto and
# long-horizon figures, and replace long prose boxes that can exceed text width.
# ---------------------------------------------------------------------------
replace_once(
    "report/sections/09_experiment_15a_fmnist_cnn.tex",
    r"""The parameter counts are
\begin{center}
\begin{tabular}{lrr}
\toprule
Tensor & Shape contribution & Parameters\\
\midrule
Conv1 kernels + bias & $8\cdot1\cdot5\cdot5+8$ & 208\\
Conv2 kernels + bias & $16\cdot8\cdot3\cdot3+16$ & 1168\\
Dense hidden + bias & $400\cdot32+32$ & 12832\\
Output + bias & $32\cdot10+10$ & 330\\
\midrule
Total & & $\mathbf{14538}$\\
\bottomrule
\end{tabular}
\end{center}""",
    r"""The parameter counts are summarized in Table~\ref{tab:exp15a-parameters}.
\begin{table}[ht]
\centering
\small
\caption{Experiment~15A compact-CNN parameterization.}
\label{tab:exp15a-parameters}
\begin{tabular}{lrr}
\toprule
Tensor & Shape contribution & Parameters\\
\midrule
Conv1 kernels + bias & $8\cdot1\cdot5\cdot5+8$ & 208\\
Conv2 kernels + bias & $16\cdot8\cdot3\cdot3+16$ & 1168\\
Dense hidden + bias & $400\cdot32+32$ & 12832\\
Output + bias & $32\cdot10+10$ & 330\\
\midrule
Total & & $\mathbf{14538}$\\
\bottomrule
\end{tabular}
\end{table}""",
)

replace_once(
    "report/sections/09_experiment_15a_fmnist_cnn.tex",
    r"""The representative comparison is
\begin{center}
\begin{tabular}{lrrrrr}
\toprule
Method & Test CE & Accuracy & Worst class & Whole train obj. & Payload\\
\midrule
Events & \textbf{0.5528} & \textbf{0.8041} & \textbf{0.460} & 0.9030 & \textbf{17.55 Mbit}\\
EF-TopK $0.5\%$ & 0.6658 & 0.7619 & 0.198 & 1.1411 & 8.07 Mbit\\
EF-TopK $2.5\%$ & 0.6075 & 0.7742 & 0.418 & 0.9795 & 40.14 Mbit\\
Dense SGD & 0.5715 & 0.7876 & 0.308 & 0.9061 & 1118.38 Mbit\\
\bottomrule
\end{tabular}
\end{center}""",
    r"""The representative comparison is summarized in Table~\ref{tab:exp15a-primary-650}.
\begin{table}[ht]
\centering
\small
\caption{Experiment~15A primary compact-CNN comparison at 650 ticks.}
\label{tab:exp15a-primary-650}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lrrrrr}
\toprule
Method & Test CE & Accuracy & Worst class & Whole train obj. & Payload\\
\midrule
Events & \textbf{0.5528} & \textbf{0.8041} & \textbf{0.460} & 0.9030 & \textbf{17.55 Mbit}\\
EF-TopK $0.5\%$ & 0.6658 & 0.7619 & 0.198 & 1.1411 & 8.07 Mbit\\
EF-TopK $2.5\%$ & 0.6075 & 0.7742 & 0.418 & 0.9795 & 40.14 Mbit\\
Dense SGD & 0.5715 & 0.7876 & 0.308 & 0.9061 & 1118.38 Mbit\\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

Figure~\ref{fig:exp15a-pareto} shows the corresponding communication--accuracy frontier on a logarithmic communication axis.

\ExpFifteenAParetoFigure""",
)

replace_once(
    "report/sections/09_experiment_15a_fmnist_cnn.tex",
    r"""The selected 650-tick event rule was frozen and evaluated on independent strong-skew partitions. The results are
\begin{center}
\begin{tabular}{llrrrr}
\toprule
Regime / seed & Method & Test CE & Accuracy & Worst class & Payload\\
\midrule
Strong / 2500 & Events & \textbf{0.5722} & \textbf{0.7946} & \textbf{0.381} & 17.21 Mbit\\
 & EF-TopK $2.5\%$ & 0.6142 & 0.7672 & 0.380 & 40.14 Mbit\\
 & Dense & 0.6291 & 0.7496 & 0.225 & 1118.38 Mbit\\
\addlinespace
Strong / 2600 & Events & \textbf{0.5813} & \textbf{0.7868} & \textbf{0.442} & 17.75 Mbit\\
 & EF-TopK $2.5\%$ & 0.6263 & 0.7680 & 0.238 & 40.14 Mbit\\
 & Dense & 0.6756 & 0.7577 & 0.186 & 1118.38 Mbit\\
\bottomrule
\end{tabular}
\end{center}""",
    r"""The selected 650-tick event rule was frozen and evaluated on independent strong-skew partitions. Table~\ref{tab:exp15a-strong-robustness} reports the results.
\begin{table}[ht]
\centering
\small
\caption{Experiment~15A robustness across independent strong-skew partitions at 650 ticks.}
\label{tab:exp15a-strong-robustness}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llrrrr}
\toprule
Regime / seed & Method & Test CE & Accuracy & Worst class & Payload\\
\midrule
Strong / 2500 & Events & \textbf{0.5722} & \textbf{0.7946} & \textbf{0.381} & 17.21 Mbit\\
 & EF-TopK $2.5\%$ & 0.6142 & 0.7672 & 0.380 & 40.14 Mbit\\
 & Dense & 0.6291 & 0.7496 & 0.225 & 1118.38 Mbit\\
\addlinespace
Strong / 2600 & Events & \textbf{0.5813} & \textbf{0.7868} & \textbf{0.442} & 17.75 Mbit\\
 & EF-TopK $2.5\%$ & 0.6263 & 0.7680 & 0.238 & 40.14 Mbit\\
 & Dense & 0.6756 & 0.7577 & 0.186 & 1118.38 Mbit\\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}""",
)

replace_once(
    "report/sections/09_experiment_15a_fmnist_cnn.tex",
    r"""The same frozen event rule was also transferred to IID and moderate label skew:
\begin{center}
\begin{tabular}{llrrrr}
\toprule
Regime & Method & Test CE & Accuracy & Worst class & Payload\\
\midrule
IID & Events & 0.5782 & 0.7916 & \textbf{0.498} & \textbf{6.55 Mbit}\\
 & EF-TopK $2.5\%$ & 0.5831 & 0.7855 & 0.304 & 40.14 Mbit\\
 & Dense & \textbf{0.5576} & \textbf{0.7965} & 0.473 & 1118.38 Mbit\\
\addlinespace
Moderate & Events & \textbf{0.5426} & \textbf{0.8059} & \textbf{0.551} & \textbf{12.45 Mbit}\\
 & EF-TopK $2.5\%$ & 0.5956 & 0.7802 & 0.320 & 40.14 Mbit\\
 & Dense & 0.5551 & 0.7974 & 0.458 & 1118.38 Mbit\\
\bottomrule
\end{tabular}
\end{center}""",
    r"""The same frozen event rule was also transferred to IID and moderate label skew, as summarized in Table~\ref{tab:exp15a-heterogeneity-transfer}.
\begin{table}[ht]
\centering
\small
\caption{Experiment~15A transfer of the frozen 650-tick event rule to IID and moderate label skew.}
\label{tab:exp15a-heterogeneity-transfer}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llrrrr}
\toprule
Regime & Method & Test CE & Accuracy & Worst class & Payload\\
\midrule
IID & Events & 0.5782 & 0.7916 & \textbf{0.498} & \textbf{6.55 Mbit}\\
 & EF-TopK $2.5\%$ & 0.5831 & 0.7855 & 0.304 & 40.14 Mbit\\
 & Dense & \textbf{0.5576} & \textbf{0.7965} & 0.473 & 1118.38 Mbit\\
\addlinespace
Moderate & Events & \textbf{0.5426} & \textbf{0.8059} & \textbf{0.551} & \textbf{12.45 Mbit}\\
 & EF-TopK $2.5\%$ & 0.5956 & 0.7802 & 0.320 & 40.14 Mbit\\
 & Dense & 0.5551 & 0.7974 & 0.458 & 1118.38 Mbit\\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}""",
)

replace_once(
    "report/sections/09_experiment_15a_fmnist_cnn.tex",
    r"""This result should be retained explicitly:
\begin{equation}
\boxed{\text{a jump schedule that is strong at 650 ticks need not remain stable at 1200 ticks}.}
\end{equation}""",
    r"""This result should be retained explicitly.
\begin{takeawaybox}[title={Experiment 15A negative result}]
A jump schedule that is strong at 650 ticks need not remain stable at 1200 ticks. Medium-horizon tuning is therefore insufficient for the event-update resolution.
\end{takeawaybox}""",
)

replace_once(
    "report/sections/09_experiment_15a_fmnist_cnn.tex",
    r"""The key 1200-tick configurations are
\begin{center}
\begin{tabular}{lrrrrr}
\toprule
$(q_0,p)$ & Train obj. & Test CE & Accuracy & Worst class & Payload\\
\midrule
$(0.0075,0.1)$ & 0.5879 & 0.6078 & 0.7808 & 0.407 & 31.96 Mbit\\
$(0.0075,0.3)$ & 0.5535 & 0.5696 & 0.7930 & 0.338 & 32.86 Mbit\\
$(0.0075,0.5)$ & 0.5624 & 0.5795 & 0.7897 & 0.406 & 34.56 Mbit\\
$(0.01,0.3)$ & 0.5960 & 0.6046 & 0.7786 & 0.468 & 28.83 Mbit\\
$\mathbf{(0.01,0.5)}$ & \textbf{0.5442} & \textbf{0.5648} & \textbf{0.7973} & \textbf{0.514} & 30.34 Mbit\\
$(0.0125,0.5)$ & 0.5828 & 0.5858 & 0.7877 & 0.316 & 27.79 Mbit\\
\bottomrule
\end{tabular}
\end{center}""",
    r"""The key 1200-tick configurations are summarized in Table~\ref{tab:exp15a-resolution-sweep}.
\begin{table}[ht]
\centering
\small
\caption{Experiment~15A long-horizon event-resolution schedule refinement.}
\label{tab:exp15a-resolution-sweep}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lrrrrr}
\toprule
$(q_0,p)$ & Train obj. & Test CE & Accuracy & Worst class & Payload\\
\midrule
$(0.0075,0.1)$ & 0.5879 & 0.6078 & 0.7808 & 0.407 & 31.96 Mbit\\
$(0.0075,0.3)$ & 0.5535 & 0.5696 & 0.7930 & 0.338 & 32.86 Mbit\\
$(0.0075,0.5)$ & 0.5624 & 0.5795 & 0.7897 & 0.406 & 34.56 Mbit\\
$(0.01,0.3)$ & 0.5960 & 0.6046 & 0.7786 & 0.468 & 28.83 Mbit\\
$\mathbf{(0.01,0.5)}$ & \textbf{0.5442} & \textbf{0.5648} & \textbf{0.7973} & \textbf{0.514} & 30.34 Mbit\\
$(0.0125,0.5)$ & 0.5828 & 0.5858 & 0.7877 & 0.316 & 27.79 Mbit\\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}""",
)

replace_once(
    "report/sections/09_experiment_15a_fmnist_cnn.tex",
    r"""The central mechanism conclusion is
\begin{equation}
\boxed{\text{late CNN degradation is a jump-resolution problem, not a demonstrated event-learning floor}.}
\end{equation}""",
    r"""The central mechanism conclusion is summarized as follows.
\begin{takeawaybox}[title={Resolution-annealing conclusion}]
The observed late CNN degradation is a jump-resolution problem, not a demonstrated event-learning floor. Persistent evidence can remain useful while the applied model quantum requires stronger late-stage refinement.
\end{takeawaybox}""",
)

replace_once(
    "report/sections/09_experiment_15a_fmnist_cnn.tex",
    r"""The canonical same-run comparison is
\begin{center}
\begin{tabular}{lrrrrr}
\toprule
Method & Test CE & Accuracy & Worst class & Whole train obj. & Payload\\
\midrule
Events & 0.5648 & \textbf{0.7973} & 0.514 & \textbf{0.7952} & \textbf{30.34 Mbit}\\
EF-TopK $2.5\%$ & \textbf{0.5527} & 0.7892 & 0.381 & 0.8337 & 74.14 Mbit\\
Dense SGD & 0.5785 & 0.7799 & \textbf{0.552} & 0.8245 & 2065.56 Mbit\\
\bottomrule
\end{tabular}
\end{center}""",
    r"""The canonical same-run comparison is summarized in Table~\ref{tab:exp15a-canonical-1200}.
\begin{table}[ht]
\centering
\small
\caption{Experiment~15A canonical paired 1200-tick validation in one fixed numerical environment.}
\label{tab:exp15a-canonical-1200}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lrrrrr}
\toprule
Method & Test CE & Accuracy & Worst class & Whole train obj. & Payload\\
\midrule
Events & 0.5648 & \textbf{0.7973} & 0.514 & \textbf{0.7952} & \textbf{30.34 Mbit}\\
EF-TopK $2.5\%$ & \textbf{0.5527} & 0.7892 & 0.381 & 0.8337 & 74.14 Mbit\\
Dense SGD & 0.5785 & 0.7799 & \textbf{0.552} & 0.8245 & 2065.56 Mbit\\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

Figure~\ref{fig:exp15a-final} visualizes the final long-horizon communication--accuracy operating points after the resolution schedule is corrected.

\ExpFifteenAFinalFigure""",
)

replace_once(
    "report/sections/09_experiment_15a_fmnist_cnn.tex",
    r"""The appropriate claim is therefore a Pareto result rather than universal metric dominance:
\begin{equation}
\boxed{\text{compact CNN event learning attains competitive predictive performance at much lower communication}.}
\end{equation}""",
    r"""The appropriate claim is therefore a Pareto result rather than universal metric dominance.
\begin{takeawaybox}[title={Experiment 15A main result}]
Compact-CNN event learning attains competitive predictive performance at much lower communication. EF-TopK or dense training can win individual predictive metrics, but neither matches the selected event operating point at comparable payload.
\end{takeawaybox}""",
)

replace_once(
    "report/sections/09_experiment_15a_fmnist_cnn.tex",
    r"""The empirical progression now reaches
\begin{equation}
\boxed{
\text{quadratics}
\rightarrow
\text{convex logistic regression}
\rightarrow
\text{nonconvex MLPs}
\rightarrow
\text{real-data MLPs}
\rightarrow
\text{convolutional neural networks}.
}
\end{equation}
Across this progression, the basic communication mechanism remains recognizable:
\begin{equation}
\boxed{
\text{local stochastic gradient}
\rightarrow
\text{temporal evidence accumulation}
\rightarrow
\text{threshold crossing}
\rightarrow
\text{sparse signed coordinate update}.
}
\end{equation}""",
    r"""The empirical progression now reaches quadratics, convex logistic regression, nonconvex MLPs, real-data MLPs, and finally convolutional neural networks.
\begin{takeawaybox}[title={Empirical progression through Experiment 15A}]
Across this progression, the recognizable communication mechanism remains: local learning signal $\rightarrow$ temporal evidence accumulation $\rightarrow$ threshold crossing $\rightarrow$ sparse signed coordinate update. The CNN result therefore extends the mechanism to a new architecture class without requiring filter-specific communication machinery.
\end{takeawaybox}""",
)

# ---------------------------------------------------------------------------
# Q1: the structural table is wide and two long boxed prose claims are replaced
# by wrapping takeaways.
# ---------------------------------------------------------------------------
replace_once(
    "report/sections/10b_novelty_equivalence_audit.tex",
    r"""This prior work establishes that the following combination is \emph{not} new:
\begin{equation}
\boxed{
\text{per-coordinate temporal gradient accumulation}
+\text{threshold crossing}
+\text{address/sign communication}
+\text{asynchronous sparse model updates}.
}
\end{equation}""",
    r"""This prior work establishes that the broad mechanism cannot be claimed as new.
\begin{takeawaybox}[title={Q1 prior-art boundary}]
Per-coordinate temporal gradient accumulation, threshold crossing, address/sign communication, and asynchronous sparse model updates already exist in closely related residual-pulse methods. The novelty question must therefore concern the complete V1 operator, not these ingredients individually.
\end{takeawaybox}""",
)

replace_once(
    "report/sections/10b_novelty_equivalence_audit.tex",
    r"""\begin{tabular}{p{3.6cm}ccccc}""",
    r"""\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{p{3.6cm}ccccc}""",
)
replace_once(
    "report/sections/10b_novelty_equivalence_audit.tex",
    r"""\bottomrule
\end{tabular}
\end{table}

The table makes the novelty boundary explicit.""",
    r"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

Table~\ref{tab:q1-structure} makes the novelty boundary explicit.""",
)
replace_once(
    "report/sections/10b_novelty_equivalence_audit.tex",
    r"""Its candidate distinction is the specific \emph{lossy pulse encoder}
\begin{equation}
\boxed{
\text{leaky coordinate evidence}
\rightarrow
\text{first passage}
\rightarrow
\text{sign/address pulse of independent size }q_r
\rightarrow
\text{full reset},
}
\end{equation}
used as a federated client-update communication layer.""",
    r"""Its candidate distinction is the specific lossy pulse encoder used as a federated client-update communication layer.
\begin{takeawaybox}[title={Q1 candidate distinction}]
V1 combines leaky coordinate evidence, first-passage triggering, a sign/address pulse with independently controlled model quantum $q_r$, and full reset of fired evidence. This complete operator---rather than accumulation, thresholding, or sign coding alone---is the object carried into the final FL experiments.
\end{takeawaybox}""",
)

# ---------------------------------------------------------------------------
# Q2: make the wide communication table width-safe, add the communication
# breakdown figure, and convert long prose boxes to wrapping takeaways.
# ---------------------------------------------------------------------------
replace_once(
    "report/sections/10d_bidirectional_protocol.tex",
    r"""Q2 therefore asks
\begin{equation}
\boxed{
\text{Does the communication advantage survive explicit server--client synchronization?}
}
\end{equation}""",
    r"""Q2 therefore asks the following end-to-end systems question.
\begin{takeawaybox}[title={Q2 question}]
Does the communication advantage survive explicit server--client synchronization once downlink traffic and client catch-up are accounted for?
\end{takeawaybox}""",
)
replace_once(
    "report/sections/10d_bidirectional_protocol.tex",
    r"""\begin{tabular}{llrrrrr}
\toprule
Uplink method & Downlink""",
    r"""\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llrrrrr}
\toprule
Uplink method & Downlink""",
)
replace_once(
    "report/sections/10d_bidirectional_protocol.tex",
    r"""Dense & Dense sync & 0.5785 & 77.99 & 2065.84 & 2070.17 & 4136.02\\
\bottomrule
\end{tabular}
\end{table}""",
    r"""Dense & Dense sync & 0.5785 & 77.99 & 2065.84 & 2070.17 & 4136.02\\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

Figure~\ref{fig:q2-bidirectional} compares the conservative hybrid V1 and EF protocols with dense bidirectional synchronization and makes the downlink contribution explicit.

\QTwoCommunicationFigure""",
)
replace_once(
    "report/sections/10d_bidirectional_protocol.tex",
    r"""Therefore
\begin{equation}
\boxed{
\text{uplink compression alone is insufficient for a strong end-to-end claim.}
}
\end{equation}
A sparse global-model synchronization mechanism is essential.""",
    r"""Therefore the protocol result must be interpreted end to end.
\begin{takeawaybox}[title={Q2 systems lesson}]
Uplink compression alone is insufficient for a strong end-to-end claim. A sparse global-model synchronization mechanism is essential because dense downlink traffic almost completely masks the difference between sparse uplink methods.
\end{takeawaybox}""",
)
replace_once(
    "report/sections/10d_bidirectional_protocol.tex",
    r"""This verdict has an important qualification. The strong end-to-end claim is supported by the \emph{combined} architecture
\begin{equation}
\boxed{
\text{sparse V1 uplink}
+\text{sparse ordered server-update log}
+\text{checkpoint fallback}.
}
\end{equation}""",
    r"""This verdict has an important qualification.
\begin{takeawaybox}[title={Q2 final protocol architecture}]
The strong end-to-end claim is supported by the combined architecture of sparse V1 uplink, an ordered sparse server-update log for exact replay, and dense checkpoint fallback when replay becomes more expensive.
\end{takeawaybox}""",
)

# ---------------------------------------------------------------------------
# Q3: width-safe main table, add the local-work figure, and replace the main
# compatibility box with a wrapping statement.
# ---------------------------------------------------------------------------
replace_once(
    "report/sections/10c_fedavg_compatibility.tex",
    r"""\begin{tabular}{crrrrr}
\toprule
$E$ & Method""",
    r"""\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{crrrrr}
\toprule
$E$ & Method""",
)
replace_once(
    "report/sections/10c_fedavg_compatibility.tex",
    r"""10 & Sign-EF & 0.4690 & 82.80 & 57.2 & 38.78\\
\bottomrule
\end{tabular}
\end{table}""",
    r"""10 & Sign-EF & 0.4690 & 82.80 & 57.2 & 38.78\\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

Figure~\ref{fig:q3-local-work} visualizes the central Q3 trend: the one-step bridge is weaker, while Event-FedAvg becomes fully competitive once the client performs genuine multi-step local training.

\QThreeLocalWorkFigure""",
)
replace_once(
    "report/sections/10c_fedavg_compatibility.tex",
    r"""Consequently the supported claim is
\begin{equation}
\boxed{
\text{V1 is compatible with conventional multi-step FedAvg local updates.}
}
\end{equation}""",
    r"""Consequently the supported claim is deliberately limited to compatibility rather than universal dominance.
\begin{takeawaybox}[title={Q3 compatibility conclusion}]
V1 is compatible with conventional multi-step FedAvg local updates. The evidence does not support the stronger claim that one fixed event setting dominates dense FedAvg for every amount of local work and every horizon.
\end{takeawaybox}""",
)

# ---------------------------------------------------------------------------
# Final matched baseline campaign: add the publication-facing frontier figure,
# make the widest traffic table safe, and replace the long boxed trade-off.
# ---------------------------------------------------------------------------
replace_once(
    "report/sections/10f_final_matched_baseline_campaign.tex",
    r"""\end{table}

The CNN result is intentionally not summarized as uniform dominance.""",
    r"""\end{table}

Figure~\ref{fig:final-baseline-frontier} places the held-out MLP and CNN results on the same communication--accuracy view and makes the final Pareto interpretation visually explicit.

\FinalBaselineFigure

The CNN result is intentionally not summarized as uniform dominance.""",
)
replace_once(
    "report/sections/10f_final_matched_baseline_campaign.tex",
    r"""\begin{tabular}{llrrrr}
\toprule
Architecture & Method""",
    r"""\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llrrrr}
\toprule
Architecture & Method""",
)
replace_once(
    "report/sections/10f_final_matched_baseline_campaign.tex",
    r"""CNN & EF-TopK 1\% & 0.6590$\pm$0.0031 & 75.83$\pm$0.26 & 19.4$\pm$1.6 & \textbf{63.9$\pm$0.0}\\
\bottomrule
\end{tabular}
\end{table}""",
    r"""CNN & EF-TopK 1\% & 0.6590$\pm$0.0031 & 75.83$\pm$0.26 & 19.4$\pm$1.6 & \textbf{63.9$\pm$0.0}\\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}""",
)
replace_once(
    "report/sections/10f_final_matched_baseline_campaign.tex",
    r"""The quality-selected and traffic-matched Strom results together therefore expose a clear trade-off:
\begin{equation}
\boxed{
\begin{array}{c}
\text{residual-conserving tied pulses can buy slightly better CNN accuracy}\\
\text{only by moving far toward the high-traffic side of the frontier,}\\
\text{whereas at comparable traffic V1 gives the stronger predictive operating point.}
\end{array}}
\end{equation}""",
    r"""The quality-selected and traffic-matched Strom results together therefore expose a clear trade-off.
\begin{takeawaybox}[title={Final nearest-neighbor result}]
Residual-conserving tied pulses can buy slightly better CNN accuracy only by moving far toward the high-traffic side of the frontier. At comparable communication, V1 gives the stronger predictive operating point on both tested architectures.
\end{takeawaybox}""",
)

# ---------------------------------------------------------------------------
# Final decision: the long one-line boxed conclusion is replaced by a width-safe
# callout. The existing fbox/parbox at the end is already width constrained.
# ---------------------------------------------------------------------------
replace_once(
    "report/sections/10e_final_algorithm_decision.tex",
    r"""The combined evidence supports the following overall conclusion:
\begin{equation}
\boxed{
\text{The present evidence supports a complete communication-efficient FL algorithm candidate.}
}
\end{equation}""",
    r"""The combined evidence supports the following overall conclusion.
\begin{takeawaybox}[title={Consolidated algorithm decision}]
The present evidence supports a complete communication-efficient FL algorithm candidate. The remaining qualification concerns the breadth of the novelty claim rather than empirical viability, FedAvg compatibility, or protocol completeness.
\end{takeawaybox}""",
)

print("Report readability patch applied successfully.")
