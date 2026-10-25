import unittest
import numpy as np
from geometry import pixels_to_ground, measure_width


class GeometryTests(unittest.TestCase):
    k = np.array([[1500., 0., 920.], [0., 1500., 2048.], [0., 0., 1.]])
    h = 1.24

    def camera(self, pitch, roll):
        p, r = np.radians([pitch, roll])
        rx = np.array([[1,0,0], [0,np.cos(p),-np.sin(p)], [0,np.sin(p),np.cos(p)]])
        rz = np.array([[np.cos(r),-np.sin(r),0], [np.sin(r),np.cos(r),0], [0,0,1]])
        return rz @ rx

    def project(self, x, z, rotation):
        world = np.column_stack((x, np.full(len(x), self.h), z))
        camera = world @ rotation.T
        pixels = camera @ self.k.T
        return pixels[:, :2] / pixels[:, 2, None]

    def test_roundtrip_with_pitch_roll(self):
        for pitch, roll in [(0,0), (12,7), (-5,-8), (25,15)]:
            rotation = self.camera(pitch, roll)
            up = rotation @ np.array([0.,-1.,0.])
            z = np.linspace(4,12,20)
            x = np.linspace(-.5,.5,20)
            pixels = self.project(x,z,rotation)
            ground, valid = pixels_to_ground(pixels,self.k,up,self.h)
            self.assertTrue(valid.all())
            np.testing.assert_allclose(ground,np.column_stack((x,z)),atol=1e-10)

    def test_known_width_with_angles_and_different_depth_sampling(self):
        for pitch, roll in [(0,0),(12,7),(-5,-8)]:
            rotation = self.camera(pitch,roll)
            up = rotation @ np.array([0.,-1.,0.])
            for slope in (0., .3, -.25):
                width = .9
                distance = width*np.hypot(slope,1)
                zl,zr = np.linspace(4,12,71),np.linspace(5,13,85)
                left = self.project(slope*zl-distance/2,zl,rotation)
                right = self.project(slope*zr+distance/2,zr,rotation)
                calibration = {'K':self.k,'up_camera':up,'image_size':[1840,4096]}
                result,samples = measure_width(left,right,calibration,self.h,frame_margin_px=0)
                self.assertAlmostEqual(result['width_m'],width,places=9)
                self.assertEqual(samples.shape,(50,5))

    def test_above_and_near_horizon_are_rejected(self):
        points = np.array([[920,2040],[920,2048],[920,2055],[920,2200]])
        ground,valid = pixels_to_ground(points,self.k,[0,-1,0],self.h)
        np.testing.assert_array_equal(valid,[False,False,False,True])
        self.assertTrue(np.isnan(ground[:3]).all())

    def test_empty_measurement(self):
        calibration = {'K':self.k,'up_camera':[0,-1,0],'image_size':[1840,4096]}
        result,samples = measure_width(np.empty((0,2)),np.empty((0,2)),calibration,self.h)
        self.assertIsNone(result['width_m'])
        self.assertEqual(samples.shape,(0,5))

    def test_height_is_metric_scale(self):
        rotation = self.camera(12,7)
        up = rotation @ np.array([0.,-1.,0.])
        pixels = self.project(np.array([.3]),np.array([5.]),rotation)
        first,_ = pixels_to_ground(pixels,self.k,up,self.h)
        second,_ = pixels_to_ground(pixels,self.k,up,self.h*2)
        np.testing.assert_allclose(second,first*2)


if __name__ == '__main__':
    unittest.main()
