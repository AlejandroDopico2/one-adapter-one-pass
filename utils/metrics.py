"""Accuracy-matrix persistence and benchmark metric definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean


@dataclass
class AccuracyMatrix:
    rows: list[list[float | None]] = field(default_factory=list)
    overall: list[float] = field(default_factory=list)

    def record(self, train_task: int, eval_task: int, accuracy: float) -> None:
        if not 0 <= eval_task <= train_task:
            raise ValueError("Only seen tasks may be evaluated")
        while len(self.rows) <= train_task:
            self.rows.append([None] * (len(self.rows) + 1))
        self.rows[train_task][eval_task] = accuracy

    def record_overall(self, accuracy: float) -> None:
        self.overall.append(accuracy)

    def _values(self, row: int) -> list[float]:
        values = self.rows[row][: row + 1]
        if any(value is None for value in values):
            raise ValueError(f"Incomplete accuracy row {row}")
        return [float(value) for value in values]

    def summary(self) -> dict:
        if not self.rows:
            raise ValueError("Accuracy matrix is empty")
        incremental = [mean(self._values(task)) for task in range(len(self.rows))]
        diagonal = [self._values(task)[task] for task in range(len(self.rows))]
        stability = [
            self._values(train_task)[eval_task]
            for train_task in range(1, len(self.rows))
            for eval_task in range(train_task)
        ]
        final_row = self._values(len(self.rows) - 1)
        forgetting = [
            max(
                float(self.rows[train_task][eval_task])
                for train_task in range(eval_task, len(self.rows))
            )
            - final_row[eval_task]
            for eval_task in range(len(self.rows) - 1)
        ]
        headline = self.overall if self.overall else incremental
        summary: dict[str, float | None | list[float]] = {
            "final_accuracy": headline[-1],
            "mean_incremental_accuracy": mean(headline),
            "final_task_mean_accuracy": incremental[-1],
            "plasticity": mean(diagonal),
            "stability": mean(stability) if stability else None,
            "mean_forgetting": mean(forgetting) if forgetting else 0.0,
        }
        if self.overall:
            summary["overall_accuracy"] = self.overall[-1]
            summary["overall_accuracies"] = list(self.overall)
        return summary

    def as_dict(self) -> dict:
        return {"accuracy_matrix": self.rows, **self.summary()}
