# 动态回归, 2个序列的最长公共连续子序列
'''
s = 'abcxvv'
m = 'adxxabcyyy'

'''

from copy import deepcopy


class Solution(object):

    def findSubstring(self, s, words):
        w_len, w_n = len(words[0]), len(words)
        if w_len * w_n > len(s):
            return []
        words_dict = {}
        for a in words:
            words_dict[a] = words_dict.get(a, 0) + 1
        res = []
        for add_n in range(w_len):
            w_dict = deepcopy(words_dict)
            s_list = [s[(i * w_len):((i + 1) * w_len)] for i in range(int(len(s) / w_len))]
            right_i, n_fit, all_n_fit = 0, 0, len(w_dict)
            for left_i in range(len(s_list) - w_n + 1):  # 左指针遍历
                if (left_i >= 1) and s_list[left_i - 1] in w_dict:
                    k = s_list[left_i - 1]
                    w_dict[k] += 1
                    if w_dict[k] == 0:
                        n_fit += 1
                    if w_dict[k] == -1:
                        n_fit -= 1
                while right_i < left_i + w_n:  # 右边界
                    k = s_list[right_i]
                    if k in w_dict:
                        w_dict[k] -= 1
                        if w_dict[k] == 0:
                            n_fit += 1
                        if w_dict[k] == 1:
                            n_fit -= 1
                    right_i += 1
                if n_fit == all_n_fit:
                    res.append(left_i * w_len + add_n)
        return res


def swap(arr, id1, id2):
    arr[id1], arr[id2] = arr[id2], arr[id1]


class Solution2(object):

    def nextPermutation(self, nums):
        """
        指针从后往前，若前面<后面，则交换+排序，否则

        2 2 5 3 5 4 3  -> 2 2 5 4 3 3 5
        2 2 5 3 5 4 4  -> 2 2 5 4 3 3 3

        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        n_len = len(nums)
        if n_len <= 1:
            return nums

        big_id = n_len - 1
        while big_id > 0:
            small_id = big_id - 1
            if nums[small_id] < nums[big_id]:
                while (big_id <= len(nums) - 1) and \
                        (nums[small_id] < nums[big_id]):
                    big_id += 1
                swap(nums, small_id, big_id - 1)
                nums = nums[:(small_id + 1)] + sorted(nums[(small_id + 1):])
                return nums
            else:
                big_id -= 1

        sorted(nums)
        return nums


Solution2().nextPermutation([1, 3, 2])


def quick_sort(nums):
    if len(nums) < 1:
        return []
    return quick_sort([i for i in nums[1:] if i < nums[0]]) + [nums[0]] + quick_sort(
        [i for i in nums[1:] if i >= nums[0]])


quick_sort([3, 4, 1, 4, 5, 7, 8])

