"""
Modified from https://github.com/facebookresearch/fvcore
"""
__all__ = ["Registry"]


class Registry:
    """
    注册表类。
    提供（名称 -> 对象) 映射的注册表，以支持自定义模块。

    要创建一个注册表（例如一个骨干网络注册表）：
        BACKBONE_REGISTRY = Registry('BACKBONE')
    要注册一个对象：
        @BACKBONE_REGISTRY.register()  # 装饰器
        class MyBackbone(nn.Module):
    或者：
        BACKBONE_REGISTRY.register(MyBackbone)  # 函数调用
    """

    def __init__(self, name):
        # 初始化注册表，name 是注册表的名称
        self._name = name
        # _obj_map: 存储名称到对象的映射的字典
        self._obj_map = dict()

    def _do_register(self, name, obj, force=False):
        """ 注册对象 
        如果名称已经存在于 _obj_map 中且 force 为 False(不强制)，则抛出异常
        """
        if name in self._obj_map and not force:
            raise KeyError(
                'An object named "{}" was already '
                'registered in "{}" registry'.format(name, self._name)
            )
        # 将名称和对象添加到 _obj_map 中
        self._obj_map[name] = obj

    def register(self, obj=None, force=False):
        """ 注册装饰器

        参数:
            - obj: 待注册的对象 (实例化的类或函数)
            - force: 如果为 True，则强制注册，即使名称已经存在于 _obj_map 中
        作用:
            - obj 为 None: 作为装饰器使用
            - obj 不为 None: 作为函数调用使用
        
        """
        # 如果 obj 为 None，则作为装饰器使用
        if obj is None:
            def wrapper(fn_or_class):
                # 获取函数或类的名称
                name = fn_or_class.__name__
                # 注册名称和对象
                self._do_register(name, fn_or_class, force=force)
                return fn_or_class
            return wrapper
        else: # 如果 obj 不为 None，则作为函数调用使用
            # 获取对象的名称
            name = obj.__name__
            # 注册名称和对象
            self._do_register(name, obj, force=force)

    def get(self, name):
        """ 获取被注册的对象 
        参数:
            name (str): 对象的名称
        返回:
            object: 名称对应的被注册的对象
        """
        # 如果名称不在 _obj_map 中，则抛出 KeyError 异常
        if name not in self._obj_map: # _obj_map: 存储名称到对象的映射的字典
            raise KeyError(
                'Object name "{}" does not exist '
                'in "{}" registry'.format(name, self._name)
            )
        # 返回名称对应的被注册的对象
        return self._obj_map[name]

    def registered_names(self):
        """返回所有已注册的名称列表
        返回:
            list: 所有已注册的名称 (字符串) 列表
        """
        return list(self._obj_map.keys())
