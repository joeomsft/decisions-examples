from pydantic import BaseModel, ConfigDict, Field


class TokenPricing(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    input_usd_per_million: float | None = Field(default=None, ge=0)
    output_usd_per_million: float | None = Field(default=None, ge=0)
    cached_input_usd_per_million: float | None = Field(default=None, ge=0)

    def estimate_cost(
        self, input_tokens: int, cached_input_tokens: int, output_tokens: int,
    ) -> float | None:
        if self.input_usd_per_million is None or self.output_usd_per_million is None:
            return None
        if cached_input_tokens and self.cached_input_usd_per_million is None:
            return None
        return (
            (input_tokens - cached_input_tokens) * self.input_usd_per_million
            + cached_input_tokens * (self.cached_input_usd_per_million or 0)
            + output_tokens * self.output_usd_per_million
        ) / 1_000_000
