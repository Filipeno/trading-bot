from .base import Executor, OrderResult


class LiveExecutor(Executor):
    """Placeholder for real-money execution. Intentionally inert until Phase 4.

    Swap PaperExecutor for this class ONLY after completing a full paper-trading
    validation period and adding ccxt live-order logic here.
    """

    def submit_order(self, symbol: str, side: str, size: float, price: float) -> OrderResult:
        raise NotImplementedError(
            "LiveExecutor is not implemented. "
            "Use PaperExecutor for all testing. "
            "See Phase 4 in the project spec before enabling this."
        )

    def get_balance(self) -> float:
        raise NotImplementedError

    def get_position(self, symbol: str) -> float:
        raise NotImplementedError
