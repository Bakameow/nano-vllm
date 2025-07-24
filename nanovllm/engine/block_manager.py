from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算完整 token 序列的哈希值，如果 prefix 不为 -1，则计算 prefix 和 token 序列的哈希值
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """
        分配 block_id 对应的 Block；会重置 Block 的引用计数、哈希值和 token_ids，并将该 Block 从 free_block_ids 队列中移除，添加到 used_block_ids 集合中

        Args:
            block_id (int): 要分配的 Block 的 ID

        Returns:
            Block: 分配的 Block
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """
        释放 block_id 对应的 Block；从 used_block_ids 集合中移除该 Block，并添加到 free_block_ids 队列中
        """
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """
        检查是否存在空闲的 block 可以分配给 seq 序列
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        分配 seq 序列占用的 Block；将 seq 的 token 分块为 Block，根据 hash(seq.block(i)) 检索全局 Block 哈希表，检查是否能够复用 Block 并视情况复用/分配 Block。seq 的分块情况将被保存在 seq.block_table 中
        """
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # 仅当 seq 序列的 Block_i 满时计算其 hash ID 
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            # 从 hash 表中，根据 hash(seq.block(i)) 查找是否存在可以复用的 Block 
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 如果可以复用 Block，则更新 seq 的 cached_tokens 计数，并增加 Block 的引用计数
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # [Q]明明是可以复用的情况，为什么还需要重新分配 Block？
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        先判断 seq 序列是否需要分配新的 Block，如果需要，返回是否存在空闲的 Block；
        如果 seq 的长度模 block_size 余数为 1，则需要分配新的 Block，判断逻辑为len(seq) % self.block_size == 1 

        Args:
            seq (Sequence): 要判断的 seq 序列

        Returns:
            bool: 如果需要分配新的 Block，则返回是否存在空闲的 Block
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
