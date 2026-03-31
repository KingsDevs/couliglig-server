"""add_handoff_dock_type

Revision ID: 3ddc1fad1852
Revises: 
Create Date: 2026-03-31 11:51:37.366059

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3ddc1fad1852'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add 'hand_off' to the dock_type_enum CHECK constraint on the docks table.

    SQLite does not support ALTER TABLE … ALTER COLUMN, so Alembic's batch mode
    is used: it recreates the table with the updated constraint and copies all
    existing rows across transparently.
    """
    old_enum = sa.Enum(
        "pickup", "receiver", "waiting_zone",
        name="dock_type_enum",
    )
    new_enum = sa.Enum(
        "pickup", "receiver", "waiting_zone", "hand_off",
        name="dock_type_enum",
    )

    with op.batch_alter_table("docks", recreate="always") as batch_op:
        batch_op.alter_column(
            "dock_type",
            existing_type=old_enum,
            type_=new_enum,
            existing_nullable=False,
        )


def downgrade() -> None:
    """Remove 'hand_off' from the dock_type_enum CHECK constraint.

    Any rows with dock_type='hand_off' must be removed before downgrading,
    otherwise SQLite will reject them.
    """
    old_enum = sa.Enum(
        "pickup", "receiver", "waiting_zone", "hand_off",
        name="dock_type_enum",
    )
    new_enum = sa.Enum(
        "pickup", "receiver", "waiting_zone",
        name="dock_type_enum",
    )

    with op.batch_alter_table("docks", recreate="always") as batch_op:
        batch_op.alter_column(
            "dock_type",
            existing_type=old_enum,
            type_=new_enum,
            existing_nullable=False,
        )
