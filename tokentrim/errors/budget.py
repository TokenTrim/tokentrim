from tokentrim.errors.base import TokentrimError


class BudgetExceededError(TokentrimError):
    def __init__(self, *, budget: int, actual: int) -> None:
        self.budget = budget
        self.actual = actual
        super().__init__(
            f"Token budget exceeded: actual={actual}, budget={budget}."
        )

