import { TestBed } from '@angular/core/testing';

import { SvmService } from './svm.service';

describe('SvmService', () => {
  let service: SvmService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(SvmService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
