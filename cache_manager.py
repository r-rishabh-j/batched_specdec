from transformers import DynamicCache

class CacheManager:
    def __init__(self, use_cache=True):
        self.use_cache=use_cache
        self.target_cache=None
        self.draft_cache=None
    
    def get_draft_len(self):
        if not self.draft_cache or not self.use_cache:
            return 0
        return self.draft_cache.get_seq_length()

    def get_target_len(self):
        if not self.target_cache or not self.use_cache:
            return 0
        return self.target_cache.get_seq_length()
    
    def load_target_cache(self, target_cache):
        self.target_cache = target_cache

    def load_draft_cache(self, draft_cache):
        self.draft_cache = draft_cache
    
    def crop(self, length, which='both'):
        if not self.use_cache:
            return
        assert which.lower() in ['target', 'draft', 'both']
        if which.lower() == 'target':
            self.target_cache.crop(length)
        elif which.lower() == 'draft':
            self.draft_cache.crop(length)
        else:
            self.target_cache.crop(length)
            self.draft_cache.crop(length)
    
    def evict_inactive(self, keep_mask, which='both'):
        if not self.use_cache:
            return
        assert which.lower() in ['target', 'draft', 'both']
        if which.lower() == 'target':
            self.target_cache = self.__evict(self.target_cache, keep_mask)
        elif which.lower() == 'draft':
            self.draft_cache = self.__evict(self.draft_cache, keep_mask)
        else:
            self.target_cache = self.__evict(self.target_cache, keep_mask)
            self.draft_cache = self.__evict(self.draft_cache, keep_mask)

    def __evict(self, cache, keep_mask):
        """Evict batch elements from cache where keep_mask is False. Modifies in place and returns cache."""
        if cache is None:
            return None
        
        # Early exit if nothing to evict
        if keep_mask.all():
            return cache
        
        # Old API: direct access to key_cache/value_cache lists (modify in place)
        if hasattr(cache, 'key_cache') and cache.key_cache:
            for layer_idx in range(len(cache.key_cache)):
                cache.key_cache[layer_idx] = cache.key_cache[layer_idx][keep_mask]
                cache.value_cache[layer_idx] = cache.value_cache[layer_idx][keep_mask]
            return cache
        
        # New API: use iterator and update() - must create new cache since update() appends
        new_cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(cache):
            new_cache.update(k[keep_mask], v[keep_mask], layer_idx)
        return new_cache
    
    def prune_cache_to_min(self,  start_positions):
        """ prune cache to smallest input sequence (draft cache only, target syncs separately) """
        if not self.use_cache or start_positions.numel() == 0:
            return
        min_seq_len = int(start_positions.min().item()) - 1
        self.target_cache.crop(min_seq_len)
        self.draft_cache.crop(min_seq_len)
