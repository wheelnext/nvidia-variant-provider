from __future__ import annotations

from variantlib.models.provider import VariantFeatureConfig


class NvidiaVariantPlugin:
    namespace = "nvidia"

    def _get_supposer_driver(self) -> list[str]:
        """Lookup the system to decide what `nvidia :: drivers` is locally supported.
        Returns a list of strings in order of priority."""

        # TODO
        return ["12.4", "12.3", "12.2", "12.1", "12.0"]

    def get_supported_configs(self) -> list[VariantFeatureConfig]:
        keyconfigs = []

        # Top Priority
        if (values := self._get_supposer_driver()) is not None:
            keyconfigs.append(VariantFeatureConfig(name="driver", values=values))

        return keyconfigs

    def get_all_configs(self) -> list[VariantFeatureConfig]:
        return [
            VariantFeatureConfig(
                name="driver",
                values=(
                    [f"11.{minor}" for minor in range(1, 9)]
                    + [f"12.{minor}" for minor in range(1, 9)]
                ),
            )
        ]
